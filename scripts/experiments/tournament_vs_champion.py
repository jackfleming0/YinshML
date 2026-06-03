#!/usr/bin/env python
"""Pit experiment candidate models against champion(s) — the final verdict.

CLOUD / TORCH ONLY. Loads NetworkWrapper checkpoints and plays them via the
existing ModelTournament machinery, so it needs torch and the right
encoding/value-head flags (NetworkWrapper.load_model hard-fails on a channel or
head mismatch — pass the flags the candidates were trained with).

Each candidate plays the champion with both color assignments; results are
aggregated into a candidate win-rate and an approximate Elo delta vs the
champion.

Examples:
  python scripts/experiments/tournament_vs_champion.py \
      --champion runs/champion/best.pt \
      --candidate baseline=runs_experiments/refit_v1/baseline/final.pt \
      --candidate refit_logreg=runs_experiments/refit_v1/refit_logreg/final.pt \
      --games 40 --out configs/experiments/refit_v1/tournament.json

  # derive candidate checkpoints from an experiment manifest (best-effort glob)
  python scripts/experiments/tournament_vs_champion.py \
      --champion runs/champion/best.pt \
      --from-manifest configs/experiments/refit_v1/manifest.json \
      --checkpoint-glob "**/checkpoint_iteration_*.pt" --games 40
"""

import argparse
import json
import math
import sys
from pathlib import Path


def win_rate_to_elo(win_rate: float) -> float:
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / win_rate - 1.0)


def _latest_checkpoint(save_dir: str, glob: str):
    paths = sorted(Path(save_dir).glob(glob))
    return paths[-1] if paths else None


def _resolve_candidates(args):
    candidates = {}
    for spec in args.candidate or []:
        name, path = spec.split("=", 1)
        candidates[name] = Path(path)
    if args.from_manifest:
        manifest = json.loads(Path(args.from_manifest).read_text())
        for name, arm in manifest["arms"].items():
            ckpt = _latest_checkpoint(arm["save_dir"], args.checkpoint_glob)
            if ckpt is None:
                print(f"  WARN: no checkpoint for arm {name} under {arm['save_dir']}",
                      file=sys.stderr)
            else:
                candidates.setdefault(name, ckpt)
    return candidates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", action="append", required=True,
                    help="champion checkpoint path (repeatable)")
    ap.add_argument("--candidate", action="append", metavar="NAME=CKPT",
                    help="candidate checkpoint (repeatable)")
    ap.add_argument("--from-manifest", help="experiment manifest.json to derive candidates")
    ap.add_argument("--checkpoint-glob", default="**/checkpoint_*.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-enhanced-encoding", action="store_true")
    ap.add_argument("--value-head-type", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # Imported here so --help works without torch.
    from yinsh_ml.utils.tournament import ModelTournament
    from yinsh_ml.utils.elo_manager import MatchResult  # noqa: F401  (type clarity)

    candidates = _resolve_candidates(args)
    if not candidates:
        print("no candidates resolved", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else Path("tournament_vs_champion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tourney = ModelTournament(
        training_dir=out_path.parent,
        games_per_match=max(1, args.games // 2),  # half each color assignment
        device=args.device,
        use_enhanced_encoding=args.use_enhanced_encoding,
        value_head_type=args.value_head_type,
    )

    results = {}
    for champ_path in args.champion:
        champ_id = Path(champ_path).stem
        champ = tourney._load_model(Path(champ_path))
        for name, cand_path in candidates.items():
            cand = tourney._load_model(Path(cand_path))
            # candidate as white, then as black (MatchResult fields are raw
            # win counts; total_games() is a method)
            r1 = tourney._play_match(cand, champ, name, champ_id)
            r2 = tourney._play_match(champ, cand, champ_id, name)
            cand_wins = r1.white_wins + r2.black_wins
            draws = r1.draws + r2.draws
            total = r1.total_games() + r2.total_games()
            win_rate = (cand_wins + 0.5 * draws) / total if total else 0.0
            entry = {
                "candidate": name, "champion": champ_id,
                "candidate_wins": cand_wins, "total_games": total,
                "candidate_win_rate": round(win_rate, 4),
                "elo_delta_vs_champion": round(win_rate_to_elo(win_rate), 1),
            }
            results[f"{name}_vs_{champ_id}"] = entry
            print(f"{name} vs {champ_id}: win_rate={win_rate:.3f} "
                  f"Elo~{entry['elo_delta_vs_champion']:+.0f} ({cand_wins}/{total})")

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
