# 7.1 Intrinsic-ceiling check — box runbook

**Goal:** settle whether the value head's ~0.737 AUC ceiling (see
`e25_sharkdp_value_ceiling.md` §4–5) is the **encoding** or **intrinsic/Bayes**.
Train a value net **from random init** ("from scratch") hard on a large decisive
corpus and read the peak held-out AUC.

**Verdict logic (asymmetric — read carefully):**
- **Peaks ≥ ~0.80** → even the basic encoding carries >0.74 of win/loss signal →
  iter1 (which has *strictly more* info) is limited by **trunk/training, not the
  encoding** → reopens a corpus/training angle. **Decisive.**
- **Caps ~0.74** → consistent with an **intrinsic/Bayes ceiling**, but not airtight,
  because this corpus is **6-channel** and the 15ch encoder adds score-diff + turn
  features that 6ch lacks (see "6ch caveat" below).

## Why this is a box job, not local

Four local attempts thrashed: the corpus is huge, the Mac mini has 26 GB RAM and
also serves the analysis board on the same MPS GPU, and MPS holds a large hidden
unified-memory working set. The session's repeated big loads saturated swap
(13.5/14.3 GB) and even a 0.87 GB run couldn't stay resident. A CUDA box with real
RAM + a dedicated GPU removes all of that. Partial local signal before each thrash:
from-scratch 6ch reached **test AUC ~0.68 by epoch 2 and rising** — promising, but
right in the ambiguous zone, so we need the full curve.

## Box specs

Any single-GPU CUDA box (e.g. vast.ai RTX 4090, 32–64 GB RAM). No GPU-memory
pressure — the net is small (~34M params, 6ch). This is a ~15–30 min job.

## Setup

```bash
git clone <repo> && cd YinshML
git checkout e25-binding-constraint          # the branch with --from-scratch + this runbook
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt              # torch with CUDA
```

## Data (the corpus is gitignored — transfer it)

The source `expert_games/yngine_volume.npz` (600 MB, 13.6M decisive 6ch positions)
and its subsamples are NOT in git. Two options:

**A. Quick — ship the ready-made 2M subsample (84 MB):**
```bash
# from the Mac:
scp expert_games/yngine_volume_2M_6ch.npz  user@box:~/YinshML/expert_games/
```

**B. Thorough — ship the full source (600 MB), subsample bigger on the box:**
```bash
scp expert_games/yngine_volume.npz  user@box:~/YinshML/expert_games/
# on the box (64 GB RAM handles 5M comfortably):
python scripts/subsample_npz_prefix.py --src expert_games/yngine_volume.npz \
    --out expert_games/yngine_volume_5M_6ch.npz --n 5000000
```

## Run

**1. From-scratch ceiling (the experiment):**
```bash
python scripts/value_ceiling_probe.py --from-scratch \
    --data expert_games/yngine_volume_2M_6ch.npz \
    --epochs 20 --lr 1e-3 --wd 1e-4 --batch 2048 --eval-cap 80000 --device cuda \
    | tee logs/e25_71_fromscratch.log
```
Watch the `test_auc` column climb per epoch; the **peak** is the answer. From-scratch
needs ~15–20 epochs to top out (then it overfits — train_auc → ~1.0, test declines).

**2. Control — iter1's value head on the SAME eval set** (apples-to-apples vs the
from-scratch peak; the canonical 0.737 was on the *human* corpus, this re-reads it
on yngine-held-out so the comparison is clean):
```bash
python scripts/value_ceiling_probe.py \
    --data expert_games/yngine_volume_2M_6ch.npz \
    --epochs 0 --device cuda | grep zero-shot
# (epochs 0 -> just the zero-shot eval of iter1's loaded head on this corpus)
```
If `--epochs 0` isn't honored, run with `--epochs 1` and read the `zero-shot` line.

## The 6ch caveat (don't skip)

`yngine_volume` is 6-channel; iter1 is 15-channel (6ch + 9 derived features, of which
**turn-number and score-differential are NOT recoverable from the 6ch board**). So:
- A **≥0.80** from-scratch result is decisive *upward* (basic encoding already beats
  0.74 → iter1's richer encoding can too → not an encoding/Bayes limit).
- A **~0.74** result does **not** fully rule out that 15ch's extra features add
  headroom. The clean confirmation is the **15ch upgrade** below.

## Optional 15ch upgrade (cleaner, bigger build)

To test iter1's actual encoding, regenerate a large **15ch** decisive corpus and
repeat from-scratch on it. Either re-encode yngine positions to 15ch, or generate
high-sim self-play with `scripts/gen_selfplay_labeled_corpus.py` at scale on the box
(this also feeds E26 — batch the two). Then run the same probe on the 15ch corpus.

## Report back

- From-scratch **peak test AUC** + the epoch it peaked.
- iter1 **zero-shot AUC** on the same eval set (control).
- One line: encoding (≥0.80) vs intrinsic (~0.74), with the 6ch caveat.

Append the result to `e25_sharkdp_value_ceiling.md` §7.1 and the backlog snapshot.
