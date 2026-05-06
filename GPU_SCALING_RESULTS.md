# GPU Scaling Results — 2026-05-05 4090 Sweep

> **Status**: empirical results. Companion to `GPU_SCALING_PLAN.md` (PR #11),
> which is the design-side doc. This file is the data-side: what actually
> happened when we ran the plan on a healthy 4090 box.

## TL;DR

1. **`num_workers > 0` is a real lever, but it caps fast.** Going from
   `num_workers=0` to `num_workers=4` gave a **1.85× games/hr speedup**
   (677 → 1250). Going further hurt: `num_workers=8` came in at **52% of
   peak**, `num_workers=16` at **40% of peak**. The `MAX_WORKERS=0` cap
   in `supervisor.py` was *over*-cautious; `num_workers=4` is the new
   recommended starting point on a 4090.
2. **`mcts_batch_size` between 64 and 128 is a wash at every worker
   count we tested.** Per-worker trees can't supply 128-leaf batches
   often enough for the larger size to matter. Drop the doc's "try
   64-128" recommendation; 64 is fine.
3. **The doc's "GPU low + CPU high" diagnosis was directionally right
   but wrong on mechanism.** `sm_p95` hits **73-75% from `num_workers=4`
   onwards** — when the GPU is fed, it processes at near-full speed.
   `sm_avg` stays at **10-15%** because between bursts it's idle. The
   bottleneck is not "CPU starves the GPU"; it's that **N independent
   per-worker `predict_batch` calls serialize on the GPU's command
   queue** instead of coalescing. More workers → more contention, not
   more throughput.
4. **The `MAX_WORKERS=0` "zombie process" justification was backwards.**
   Serial mode (`num_workers=0`) leaks RSS *faster* than the worker
   pool: +695-884 MB per iteration vs. +25-30 MB at `num_workers=4`.
   The supervisor process is the leakier of the two; workers actually
   clean up after themselves. Comment in `_compute_num_workers` updated
   to reflect this.
5. **The shared `BatchedEvaluator` (PR #11 Part 1 future work) is now
   the only meaningful next lever.** See `BATCHED_EVALUATOR_DESIGN.md`.

## The data

Two sweeps. Both used `configs/cloud_smoke.yaml` as the base
(48 sims/move early, 32 late, 10 games/iteration, 1 iteration), on a
single RTX 4090 (560.35.03 driver, torch 2.7.1+cu126).

### Sweep 1 — does `num_workers > 0` help at all?

| workers | batch | games/hr | sm_avg | sm_p95 | rss_growth | wall(s) |
|---|---|---|---|---|---|---|
| 0 | 64  | 677  | 8.7  | 12 | +695 MB | 71.4 |
| 0 | 128 | 695  | 8.6  | 15 | +884 MB | 70.9 |
| 4 | 64  | **1295** | 15.7 | 44 | +45 MB  | 46.3 |
| 4 | 128 | 1281 | 14.8 | 45 | +228 MB | 46.7 |

Verdict: yes, very. 1.85-1.9× speedup just from setting
`num_workers: 4`. The serial-path RSS growth was the surprise — it's
*larger* than the parallel-path growth, contradicting the historical
"workers leak as zombies" claim.

### Sweep 2 — does scaling workers further keep helping?

| workers | batch | games/hr | sm_avg | sm_p95 | rss_peak | wall(s) |
|---|---|---|---|---|---|---|
| 4  | 64  | **1250** | 15.2 | 45 | 6.5 GB  | 47.8 |
| 4  | 128 | 1225 | 15.7 | 73 | 6.4 GB  | 48.1 |
| 8  | 64  | 654  | 10.9 | 74 | 11.1 GB | 73.5 |
| 8  | 128 | 615  | 11.9 | 72 | 11.1 GB | 76.4 |
| 16 | 64  | 502  | 10.4 | 75 | 13.5 GB | 90.7 |
| 16 | 128 | 572  | 11.9 | 73 | 13.5 GB | 81.4 |

Verdict: no. Peak at 4 workers; everything past that is worse than
`num_workers=4` — *worse than the optimum, not just plateauing*. The
`rss_growth` column went mildly negative (-1.5 to -2 GB) at higher
worker counts because torch's caching allocator releases between cells
and the harness samples around that boundary. Cosmetic; ignore. The
important RSS column is `rss_peak`, which scales linearly with workers
(~2-3 GB per additional 4 workers) — that's the model + state copies.

## What this implies for `cloud_smoke.yaml` and config defaults

Updated in this PR:

```yaml
# configs/cloud_smoke.yaml
self_play:
  num_workers: 4   # was 6 — sweep showed 4 is the peak on this hardware
```

Rationale: `num_workers: 6` was untested. The data says 4 is the
peak, 8 is already 52% of peak. Six is presumably somewhere on the
downslope.

## Cost-of-throughput

At ~$0.50/hr for a 4090 cloud instance and 1250 games/hr at
`num_workers: 4`: **2,500 games per dollar**. That's the games/$/hr
metric the doc flagged as the one that actually matters. Worth
re-running the sweep on a Lambda or Vast.ai box to compare — current
data is from a single provider.

## What to do next

In rough priority:

1. **Build the shared `BatchedEvaluator`.** Detailed in
   `BATCHED_EVALUATOR_DESIGN.md`. Expected upside based on the
   `sm_p95` (73%) vs `sm_avg` (15%) gap: 3-5× games/hr ceiling.
2. **Re-run this sweep on Lambda/RunPod to validate the games/$/hr
   number.** Different host, different cost, different image — same
   sweep harness, same base config, comparable results. ~$1 of
   compute, half an hour of wall time.
3. **Audit other `*.yaml` configs in `configs/`.** Any that set
   `num_workers > 4` without measurement on this hardware are likely
   underperforming. Anything that doesn't set `num_workers` at all is
   silently running serial because of the `MAX_WORKERS=0` default.
4. **Drop the "try `mcts_batch_size: 64-128`" recommendation in
   `GPU_SCALING_PLAN.md` Part 2.** 64 is fine; 128 doesn't help at
   any worker count we tested.

## How to reproduce

On a 4090 box with CUDA + the matching torch wheel:

```bash
git checkout gpu-scaling-harness
pip install -r requirements.txt && pip install -e .

# 5-second sanity that GPU inference works at all
python scripts/gpu_probe.py

# Sweep 1 + 2 (~30min total)
python scripts/gpu_scaling_sweep.py \
    --base-config configs/cloud_smoke.yaml \
    --output-dir results/sweep_baseline_$(date +%Y%m%d_%H%M%S) \
    --workers 0 4 \
    --batch-sizes 64 128

python scripts/gpu_scaling_sweep.py \
    --base-config configs/cloud_smoke.yaml \
    --output-dir results/sweep_scale_$(date +%Y%m%d_%H%M%S) \
    --workers 4 8 16 \
    --batch-sizes 64 128
```

`gpu_probe.py` exit criteria for "host is healthy enough to bother":
`first param device: cuda:0`, build < 5s, batch-64 < 15ms/call. If any
of those fail, switch hosts before running the sweep — see the
"Multi-tenant host trap" note below.

## Multi-tenant host trap

The first cloud instance we tried showed `cuda available: True`,
matching driver, matching torch wheel — and **246-second model build,
73 ms/single-state forward, 481 ms/batch-64**. That's 50-100× slower
than bare metal. Hardware identified as a real 4090 (sm_89), but the
host was multi-tenant with a paravirtualization layer that added
huge per-kernel-launch overhead. Not fixable from inside the VM.

Cudo Compute, RunPod *Community* Cloud, and unverified Vast.ai hosts
are common offenders. RunPod *Secure* Cloud, Lambda Cloud, and
Vast.ai's verified-host listings are safer. The probe above is the
fastest way to detect the trap before burning sweep time.

---

## Threaded `BatchedEvaluator`: results vs. prediction (2026-05-06)

> **Verdict**: the threaded shared-evaluator path lands at correctness
> and robustness, but **does not deliver the predicted speedup**. On
> the codebase as it stands, GIL contention in MCTS Python code makes
> threads a net loss vs the serial path. Documented here so the next
> person doesn't re-derive it.

### What we predicted

`BATCHED_EVALUATOR_DESIGN.md` and `IMPLEMENTATION_PLAN.md` Phase 3
predicted **3-5× games/hr** from coalescing inference across N MCTS
threads. The mental model: workers do CPU-side MCTS work, drain
thread does GPU work, GPU usage rises from 15% to 50-70%.

### What we measured

Configuration: `cloud_smoke.yaml` (48 sims/move early, 32 late, 10
games/iter, 1 iteration), EPYC 7763 (128 cores) + RTX 4090,
driver 560.35.03 + torch 2.10.0+cu130. `gpu_probe` showed
bare-metal-class single-stream throughput (3242 positions/sec,
19.7ms/batch-64).

| config                       | games/hr | sm_avg | sm_p95 |
|---|---|---|---|
| `num_workers=0` (serial)     | **702**  | 7.9    | n/a    |
| `num_workers=1`, evaluator   | 541      | 6.4    | n/a    |
| `num_workers=4`, evaluator   | 470      | 5.5    | 11     |
| `num_workers=8`, evaluator   | 533*     | 5.1    | 17     |
| `num_workers=16`, evaluator  | 489      | 5.2    | 18     |

*`num_workers=8` measured at `max_wait_ms=10` (vs 1.0 default) — the
single tunable knob that helped, +18% over the same-w default.

### What we conclude

**Serial wins.** Adding worker threads doesn't just fail to help — it
*hurts* (470/489 vs 702). The threading and queue overhead costs more
than the GPU/CPU overlap gains. Per-worker throughput drops linearly
(`117 → 56 → 30 g/hr/worker`), the textbook signature of a serial
wall.

**The wall is the GIL.** The bitboard port moved a lot of MCTS
compute to C++, but enough remains in Python (selection loop,
valid-move enumeration, encoding, tree updates) that worker threads
can't run in parallel — they serialize on the GIL.

**The process pool isn't a clean baseline either.** On Python 3.12 +
this cloud image, the spawn-context process pool dies during
`_fixup_main_from_path` worker init before any games are generated.
We didn't get a working `num_workers > 0` baseline with processes
on this box. The 2026-05-05 sweep on the previous box (different
CPU, Python 3.10) showed `num_workers=4` at 1250 g/hr with the
process pool — that's the right number for "how fast can processes
go on a different box", not directly comparable to today's
threaded numbers.

### What this changes

- `cloud_run_v1.yaml` is now `num_workers: 0` and
  `use_shared_evaluator: false`, sized at 25 iterations × 200 games
  for ~5000 games / ~7h / ~$3.50 at the measured 702 g/hr.
- `BATCHED_EVALUATOR_DESIGN.md` got a postscript flagging the
  threaded shape as throughput-neutral on this codebase.
- The shipped evaluator is **kept** in the codebase — it's correct
  (7/7 unit tests, deterministic vs the direct path) and it's the
  *only* parallel path that runs on Python 3.12 + this image. Future
  bitboard work or a `torch.multiprocessing` rewrite may make it
  worthwhile to re-measure.

### What we'd try next, if speedup matters more

1. **More bitboard.** Profile MCTS with the threaded path on; whatever
   Python code shows up in `py-spy top` is what's holding the GIL.
   Move it to C++ and re-measure threaded throughput.
2. **`torch.multiprocessing` with shared model tensors.** Each worker
   is a process (own GIL) but reads from a shared GPU tensor for the
   network's parameters. Different shape than the threaded
   evaluator.
3. **Fix the Python 3.12 spawn issue.** Pin the launching script
   structure so `_fixup_main_from_path` works in workers, recovering
   the process-pool-with-cuda-init path that worked on the previous
   box.

None of these are in this PR. They're follow-ups, in the order above.
