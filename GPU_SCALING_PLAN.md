# GPU Scaling & Hardware Profile Plan

> **Status**: concepts / design notes. Nothing in this document has been implemented
> on this branch — it captures the thinking behind a planned refactor so the next
> person (or future self) can pick it up without re-deriving it.

## Why this doc exists

We are moving training from a Mac (MPS) dev loop to cloud GPU instances (4090-class
today, larger later). Three questions came up that are tangled together:

1. **Are we batching GPU evaluation correctly for self-play MCTS?**
2. **How do we know we're actually using the cloud GPU we're paying for?**
3. **How do we structure config so we can flip cleanly between
   "MPS-tuned dev" and "cloud-tuned production" without editing 30 YAML files?**

The rest of this doc walks through each in order. None of them is independent;
the config story only matters if the batching story is right, and the batching
story only matters if we actually push the GPU hard enough to expose its limits.

---

## Part 1 — Batched GPU evaluation: what we have, what we're missing

### The standard AlphaZero pattern

The textbook way to keep a GPU fed during MCTS self-play is:

1. Run many MCTS searches in parallel.
2. When a search reaches an unexpanded leaf, do **not** evaluate it inline.
   Push the leaf state onto a shared queue and pause that walk.
3. A single inference worker pulls states off the queue, accumulates a batch
   (e.g. 128–512), runs **one** `predict_batch` call, and returns results to
   the waiting walks.
4. Walks resume, back up values, and continue.

This is "leaf parallelization with virtual loss" — virtual loss is the hack
that prevents N parallel walks from all picking the same leaf.

### What is already implemented in this codebase

Within a **single game's search**, the pattern above already exists in
`yinsh_ml/training/self_play.py`:

- `Node.add_virtual_loss()` / `Node.remove_virtual_loss()` (around lines 113–121).
- `search_batch()` (around line 711) collects up to `mcts_batch_size` leaves per
  search before flushing them to the GPU.
- `_evaluate_and_backup_batch()` (around line 865) calls
  `NetworkWrapper.predict_batch()` (`yinsh_ml/network/wrapper.py:468`) once on a
  `(B, C, 11, 11)` tensor.

Config knobs that already exist: `use_batched_mcts: true`, `mcts_batch_size`,
`num_simulations`, `num_workers` (see e.g. `configs/post_bitboard_tuning_b2.yaml`,
`configs/cloud_smoke.yaml`).

### What is **not** implemented — the real gap

1. **Cross-game batching is absent.** Parallel games run via
   `ProcessPoolExecutor` (`self_play.py:1448`). Each worker process holds its
   *own* `NetworkWrapper` — its own copy of the model on the GPU. Two workers
   evaluating leaves at the same instant produce **two** GPU calls of size 32,
   not one of size 64. There is no shared inference server.
2. **`num_workers` defaults to 0** (`supervisor.py:2048`) — i.e. games run
   *serially* out of the box. This was set conservatively because of historical
   zombie-process memory leaks; it has not been re-evaluated since.
3. **The simpler `yinsh_ml/search/mcts.py`** (used at inference / tuning time,
   not at training time) has **no virtual loss and no batching at all**.
   `_evaluate_state()` evaluates one position per call. This is fine when
   you're searching one game at a time on a CPU; it is not fine on a 4090.

### When the gap matters

The gap from "per-worker batching" to "shared cross-worker batching" is real
work (see "Future: shared evaluator" below). It only pays off **after** we have
maxed out the existing knobs. The honest order is:

1. Turn on `num_workers > 0` and a reasonable `mcts_batch_size`.
2. Measure GPU utilization on the actual cloud instance.
3. If the GPU still isn't saturated, *then* build the shared evaluator.

Doing the refactor first is a common trap. A 4× speedup from a config flip is
much cheaper than a 4× speedup from a refactor, and you cannot tell which
remaining bottleneck is dominant until the easy one is gone.

### Future: shared `BatchedEvaluator`

When we get there, the shape is roughly:

- One inference process owns the only `NetworkWrapper` on the GPU.
- N MCTS workers (threads or processes) keep their own trees, but call
  `evaluator.evaluate(states)` instead of `network.predict_batch(states)`.
- The evaluator coalesces requests from all workers into one
  `predict_batch` call, then routes results back per-worker via reply queues.

Notes / sharp edges:

- Threads vs. processes: threads share memory cheaply but the GIL bites if MCTS
  CPU work is in pure Python. Processes need IPC (queues, shared tensors,
  `torch.multiprocessing.share_memory_()`), which has its own overhead.
- The existing per-worker `predict_batch` path becomes a fallback for the
  `num_workers == 1` case (and for offline tools that don't want the
  evaluator process running).
- This is incompatible with the current "each worker loads its own model from
  disk" pattern — model loading needs to move to the evaluator only.

---

## Part 2 — Are we actually using the GPU?

A perfectly valid response to "your GPU is starved" is: how would I know?
Before any refactor or even any config change, **measure**.

### Live measurement during a self-play run

```bash
# GPU compute %, mem, power, sampled every second
nvidia-smi dmon -s pucvmet -d 1

# Where is Python spending time? (sampling profiler, no instrumentation)
py-spy top --pid <selfplay_pid>

# Per-core CPU saturation
htop
```

What we want to see: `sm` (compute utilization) consistently >80%, power draw
near TDP (~450W on a 4090). What "GPU is starved" looks like: `sm` flickering
between 5% and 40%, power well under 200W.

The cross-tab tells us where the bottleneck is:

| GPU | CPU  | Diagnosis                                                       |
|-----|------|-----------------------------------------------------------------|
| Low | Low  | Python overhead, IPC, I/O — not compute-bound on either side.   |
| Low | High | CPU can't generate states fast enough to fill GPU batches.      |
| High| High | Healthy — both sides working.                                   |
| High| Low  | Rare; usually means batches are huge and rarely flushed.        |

### Knobs to push before writing any new code

```yaml
self_play:
  use_batched_mcts: true
  mcts_batch_size: 128      # was 32 — try 64, 128, 256
  num_workers: 8            # was 0 — try 4, 8, ncpu-2
  num_simulations: 200      # more sims/move = more GPU work per game
```

`num_workers > 0` is the single biggest lever currently unused. Caveat: the
zombie-process / memory-leak issue mentioned in `CLAUDE.md` may still be live;
watch RSS over a long run before committing to it as the new default.

`mcts_batch_size` matters less than it looks on its own — within one game the
search has to *find* that many independent leaves before it can flush, and
virtual loss limits how parallel a small tree can be. 64–128 is usually a sweet
spot; bigger just makes the last batch of each search half-empty.

### The metric that actually matters

Not GPU %. **Games per dollar per hour.**

- Log games/hour at the supervisor level.
- Compare across instance types: e.g. a $0.80/hr 4090 box generating 600
  games/hr is cheaper per game than a $3.00/hr H100 box generating 1500
  games/hr — and *much* cheaper if your CPU is the bottleneck and you're
  paying for the H100 to idle.
- Self-play is typically the dominant phase. If self-play is 80% of wall
  time and the GPU is idle for most of that, you're paying for a GPU you're
  using 20% of the time.

Right-sizing instances matters as much as right-sizing code. For this
workload at current defaults, a fat-CPU + 4090 is probably better balanced
than a thin-CPU + H100.

### Calibration as an action, not a guess

Optimal `mcts_batch_size` and `num_workers` cannot be predicted from first
principles — they depend on the instance's CPU/GPU ratio, network size, and
game length distribution. Plan: a `scripts/calibrate_hardware.py` that runs
short sweeps, picks the winners, and writes them back into the hardware
profile (see Part 3). Run once per new instance type.

---

## Part 3 — Hardware-aware configuration architecture

### Current state

- `configs/` contains ~30 flat, self-contained YAML files.
- Every config redefines everything: device, num_workers, batch sizes, LR
  schedule, eval mode, the works.
- `device: auto` exists at the top of each file and resolves via
  `select_device()` at `scripts/run_training.py:55`. That part is fine.
- Switching the *same experiment* between MPS and cloud-4090 today requires
  remembering which 6+ fields to flip and not missing one.

This works at low config count but doesn't scale. It also makes review hard:
you can't tell from a config diff whether a change is "I'm tuning the
experiment" or "I'm tuning for different hardware."

### Proposed: overlay configs

Split the three axes that are currently mixed into separate files, and merge
them at load time.

```
configs/
├── base.yaml                  # rules-of-the-game-level invariants
├── hardware/
│   ├── mps_dev.yaml           # num_workers=0, mcts_batch=16, autocast=false, ...
│   ├── cloud_4090.yaml        # num_workers=8, mcts_batch=128, autocast=true, ...
│   └── cloud_h100.yaml        # future
└── experiments/
    └── post_bitboard_b2.yaml  # only experiment-specific knobs
```

CLI:

```
python scripts/run_training.py \
    --config experiments/post_bitboard_b2.yaml \
    --hardware cloud_4090
```

Merge order: `base → hardware → experiment → CLI overrides`. The experiment
file should be **silent on hardware-axis fields**; if it sets them, that's a
smell and the validator should warn.

This is a small change to the config loader (deep-merge dicts; PyYAML plus
a `_recursive_merge` helper, on the order of 30 lines).

### Code vs. config: `DeviceCapabilities`

Some hardware adaptations don't belong in YAML at all — they're properties of
the device, not knobs to tune per experiment. Funnel them through one struct:

```python
@dataclass
class DeviceCapabilities:
    device: str                       # 'cuda' | 'mps' | 'cpu'
    supports_autocast_bf16: bool
    supports_torch_compile: bool
    supports_pin_memory: bool
    supports_channels_last: bool
    recommended_dataloader_workers: int
```

Code paths read from `caps`, not from `cfg` directly. Concretely:
`enable_autocast: true` in YAML becomes a *request* —
`if cfg.enable_autocast and caps.supports_autocast_bf16: ...` is what runs.
Same for `torch.compile`, `pin_memory`, `memory_format=channels_last`.

The rule of thumb:

> If turning a knob requires the user to know what hardware they're on,
> it doesn't belong in their experiment config.

Today these checks tend to scatter as `if device == "mps"` across the codebase.
Pulling them into one struct is the single biggest readability win available.

### Guardrails at load time

After merge, validate the resolved config against `caps` and **fail loudly** for
nonsense combinations. Examples:

- `num_workers > 1` on MPS → warn, clamp to 1 (multi-process MPS is fragile).
- `enable_autocast: true` on MPS → warn, disable (MPS bf16/fp16 is uneven).
- `mcts_batch_size > 256` on CPU-only → warn (will OOM later anyway).
- `use_batched_mcts: false` on `cloud_4090` → warn ("paying for a GPU you're
  not using").

Print the *resolved* config at startup. Five lines of `logger.info` saves an
hour of "wait, why was autocast off?"

### Migration path

We do not have to rewrite all 30 configs at once. Pragmatic order:

1. Add overlay support + `--hardware` flag. Back-compat: missing flag = no
   overlay = behave exactly as today.
2. Create `hardware/mps_dev.yaml` and `hardware/cloud_4090.yaml` by extracting
   the device-axis fields from `cloud_smoke.yaml` and the current MPS-tuned
   config.
3. New experiments use the overlay style.
4. Migrate active experiments only when already editing them. Old configs keep
   working unchanged.

### What this gets us

- Switching regimes is one flag.
- Reviewing an experiment means reading the experiment file — 10 lines, not 100.
- Adding a new instance type (H100, multi-GPU, A100 spot) means adding a file
  in `hardware/`, not auditing 30 files.
- `if device == "mps":` checks stop reproducing across the codebase because
  they live behind one capability struct.

---

## Part 4 — Logging and print hygiene

Training runs are currently very noisy. There is a mix of `print()` statements,
`logger.info()` calls, and per-game / per-simulation chatter scattered across
modules. At self-play scale (hundreds of games per iteration, with multiple
workers writing to the same stdout) this becomes both unreadable for a human
watching the run and expensive — string formatting and stdout writes are not
free in hot loops.

This is not a glamorous refactor but it pays for itself the first time you
need to debug a 12-hour run and have to grep through 5 GB of logs.

### What "clean" looks like

- **One pass to audit existing output.** Grep for every `print(` call across
  the codebase. The vast majority should become `logger.debug(...)` or be
  deleted outright. `print()` survives only in user-facing CLI tools (demos,
  scripts where the output *is* the output).
- **Consistent level discipline:**
  - `ERROR` — something is wrong and the run cannot continue cleanly.
  - `WARN` — something unexpected; run continues but a human should know.
  - `INFO` — milestone events. Iteration boundaries, eval results, phase
    transitions. Roughly one line per meaningful unit of work, not per game
    and definitely not per move.
  - `DEBUG` — everything else. Off by default in production runs.
- **A single root logger configuration**, set once at process startup
  (`scripts/run_training.py`, `run_large_scale_selfplay.py`, etc.), with
  level controlled by config / CLI flag. No per-module
  `logging.basicConfig()` calls scattered around.
- **Per-worker prefixes** for parallel self-play. When `num_workers > 0`,
  output from 8 processes interleaves into the same stdout and becomes
  unreadable. Each worker's logger should prepend `[worker-3]` (or PID) so
  lines can be filtered.

### Rate-limiting hot loops

Anything inside the MCTS simulation loop, the move-generation loop, or the
training-batch loop should not unconditionally log per iteration. Patterns
that work:

- **Per-N**: `if step % log_every == 0:` — log every Nth event.
- **Per-time**: log at most once per `log_interval_seconds` regardless of
  iteration count.
- **Aggregated summaries**: instead of "game 1 finished in 12.3s",
  "game 2 finished in 11.8s", … emit one line per iteration:
  `iter=42 games=128 mean_len=87.3 mean_time=11.9s gpu_util_avg=0.71`.

The third form is also what makes runs *parseable* later — see structured
logging below.

### Structured logging for metrics

Metric lines (loss, ELO, hit rate, games/hour, GPU util) should be emitted
in a consistent `key=value` form rather than free-form prose. This costs
nothing at write time and turns post-run analysis from "regex archaeology"
into "pandas read_csv". Concretely:

```
INFO  iter=42 phase=selfplay games=128 elapsed=12m04s games_per_hour=635
INFO  iter=42 phase=train  loss_policy=2.31 loss_value=0.184 lr=8.3e-4
INFO  iter=42 phase=eval   elo=1247 vs_prev=+18 win_rate=0.61
```

This pairs naturally with the existing dashboard and `MetricsLogger`
(`yinsh_ml/utils/metrics_logger.py`); the goal is for stdout to carry the
same information in human-readable form.

### Verbosity control

- A single `--verbose` / `--quiet` CLI flag (or `logging.level: INFO|DEBUG`
  in config) controls the root logger.
- Production runs (large self-play, overnight training) should be runnable
  at the default level without flooding the terminal.
- Dev runs can crank to DEBUG with one flag, no code edits.

### File logging in addition to stdout

Long runs should write to a rotating file (`logs/run_YYYY-MM-DD_HHMMSS.log`)
in addition to stdout. Stdout is for the human watching; the file is for
debugging the failure that happened at hour 9 of a 12-hour run.

### Migration path

Same incremental approach as the config refactor:

1. Add the centralized logger setup + per-worker prefix + verbosity flag.
   Back-compat: existing `print()` calls keep working, they just don't
   benefit.
2. Sweep one module at a time (start with `self_play.py` and `mcts.py` —
   the noisiest), converting `print` → `logger.debug`/`info` and adding
   rate limiting where appropriate.
3. Anything that survives as `print()` after the sweep is justified or
   deleted.

---

## Open questions / decisions deferred

These are not blockers for the doc but should be answered before writing code:

- **Zombie-process / memory-leak status.** `MAX_WORKERS = 0` was defensive.
  Is the underlying issue still live, or fixed by the bitboard work that
  landed since? Rerun a long self-play with `num_workers=8` and watch RSS.
- **Threads vs. processes for the future shared evaluator.** Depends on how
  much MCTS CPU work is pure-Python vs. NumPy/C++. The bitboard refactor
  should have moved a lot of it out of Python, which makes threads more
  attractive than they were a year ago.
- **Where does the supervised pretraining path fit?** It already loads its own
  encoder via `--use-enhanced-encoding`. The hardware-overlay system needs
  to extend cleanly to it, not just to `run_training.py`.
- **Calibration cadence.** Is `calibrate_hardware.py` something we run once
  per instance type, or once per code change that meaningfully shifts the
  CPU/GPU balance (e.g. encoder swap)? Likely the latter.

---

## TL;DR

1. Per-game leaf batching with virtual loss is **already built**. Cross-game
   batching is **not**. The first thing to do is turn on what exists
   (`num_workers > 0`) and measure.
2. The bottleneck question is empirical, not theoretical. `nvidia-smi dmon`
   plus `py-spy top` answers it in five minutes.
3. The metric is games/$/hr, not GPU %.
4. Splitting config into `base / hardware / experiments` overlays, plus a
   `DeviceCapabilities` struct for things that don't belong in YAML at all,
   gets us a clean MPS-dev ↔ cloud-prod toggle without sprinkling
   `if device == "mps"` across the codebase.
5. Logging hygiene — one centralized logger setup, level discipline,
   per-worker prefixes, rate-limited hot loops, structured `k=v` metric
   lines — turns unreadable training runs into something a human can
   actually watch and `grep` later.
