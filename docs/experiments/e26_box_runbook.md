# E26 — high-budget-search distillation (box runbook)

**Reaimed by E25:** distill the **search-improved policy** (MCTS visit counts ≫ raw
prior; the ablation proved policy is the binding head). Value is at an intrinsic
ceiling, so it's distilled only to keep the head anchored — not the target. There is
**no external teacher** (iter1 ≫ yngine > sharkdp), so the teacher is **iter1 searched
harder** (self-distillation): high-sim MCTS manufactures targets that exceed the
student, distillation banks them.

**Verdict (R1, the only one):** distilled net vs a FROZEN copy of iter1, H2H,
color-balanced. In-loop tournaments green-checked a known loser before — don't trust
anything but this.

## Pipeline (all 3 stages smoke-validated locally + on the box)

```
gen_distill_corpus.py   ->  e26_distill.py   ->  measure_h2h.py
 (teacher data, GPU)        (distill, GPU)       (gate vs frozen iter1)
```

## Setup (done once)
```bash
# branch + deps already on the box; iter1 transferred to models/iter1_ema_2026-05-27/
conda activate main; cd ~/YinshML && git pull
# deps: pip install -r requirements.txt (minus coremltools, which we made optional)
```

## Stage 1 — teacher data (the long pole) — use the E20 inference server
```bash
M=models/iter1_ema_2026-05-27/iter1_ema.pt
PYTHONPATH=~/YinshML PYTHONUNBUFFERED=1 nohup python -u scripts/gen_distill_corpus.py \
    --model $M --out expert_games/e26_teacher_800sim.npz \
    --games 42000 --sims 800 --workers 48 \
    --use-inference-server --inference-dtype bf16 --batch-size 64 \
    --max-positions 2000000 --checkpoint-every 50 \
    > logs/e26_gen.log 2>&1 < /dev/null &
```
- **`--use-inference-server` is the throughput path (E20):** one bf16 GPU server + N CPU
  workers, no per-worker CUDA context. Measured on the 4090 (2026-06-09, 800 sims):
  **48 workers = 2,256 positions / 3.5 min ≈ 645 pos/min ≈ 38.7k/hr** (mean coalesced
  GPU batch 133). That's **~4.7× the old `--device cuda` best** (which caps at ~8 workers
  then regresses — 138 pos/min) and dramatically faster than `--device cpu` (CPU forwards
  at 800 sims are brutal — the old long pole). At 38.7k/hr, the 2M-position target is
  **~52 h** (was ~10 days via the old GPU path).
- **Knobs:** scale `--workers` toward the core count (192-core box); `--batch-size 64`
  lets the server coalesce fat batches (the virtual-loss fix fills them). Raise `--sims`
  for sharper targets — at high sims the win grows (forward dominates, bf16 helps most).
- `--sims 800` is the teacher budget (higher = better targets; KL-to-prior grows with sims).
- Saves top-K (K=64) visit-count policy + root value per main-game position. Checkpoints
  every 50 games; stops at `--max-positions`. **Watch:** `game N/… (total P/target)`.
- Legacy `--device cpu|cuda` (per-worker model, no server) still works for small runs /
  debugging; the inference-server path is strictly better for the big Stage-1.

## Stage 2 — distill (fast, GPU)
```bash
PYTHONPATH=~/YinshML python scripts/e26_distill.py --init $M \
    --data expert_games/e26_teacher_800sim.npz --out models/e26_distilled.pt \
    --epochs 6 --lr 2e-4 --batch 1024 --value-weight 0.5 --device cuda \
    | tee logs/e26_distill.log
```
- Continues from iter1; soft-CE policy distill + scalar-value MSE (auto-detected).
- **Watch:** `pol_loss` should drop and `te_polCE` improve. Saves a raw state_dict.

## Stage 3 — H2H gate (the verdict)
```bash
# freeze a copy of iter1 as the yardstick first:
cp $M models/iter1_frozen.pt
PYTHONPATH=~/YinshML python scripts/measure_h2h.py \
    --white models/e26_distilled.pt --black models/iter1_frozen.pt \
    --white-label distilled --black-label iter1 --games 60 --output logs/e26_h2h.json
```
- Color-balanced; read distilled's score with its ±CI (n=60 ≈ ±13pp — must clear it).
- **>0.5 beyond CI → E26 worked.** ≈0.5 → no lift; the policy-distill lever didn't move it.

## Throughput benchmark (sanity-check before the big Stage-1)
Throughput is largely resolved (use `--use-inference-server`); a short run just
confirms pos/min on the current box so Stage 1 can be sized to the rental window:
```bash
PYTHONPATH=~/YinshML python scripts/gen_distill_corpus.py --model $M \
    --out /tmp/bench.npz --games 48 --sims 800 --workers 48 \
    --use-inference-server --inference-dtype bf16 --batch-size 64 \
    --max-positions 99999999 --checkpoint-every 9999    # read the "done: N positions, X.Xm" line
```
Positions/min × target ⇒ Stage-1 hours. Reference: ~645 pos/min @48w/800sim on the 4090.
Watch the server's `mean coalesced batch` in the log — climbing with `--workers` confirms
coalescing; if it plateaus and pos/min flattens, more workers won't help (see E20 runbook).

## Notes
- Single GPU (4090, 24 GB). The inference-server path needs **one** model on the GPU
  (the server), so workers are CPU-only and you scale them toward the core count without
  GPU-mem pressure — that's the whole point vs the old `--device cuda` (one context/worker).
- E20 throughput build (server + virtual-loss fix + bf16): `docs/experiments/e20_throughput_build.md`.
- Corpus is 15ch (iter1's encoding); ~7.3 GB per 1M positions in RAM (251 GB box — fine).
