#!/usr/bin/env python3
"""Dominance + playstyle of completed analysis-board games, from the engine's
side (provenance-tagged when available, else inferred from move timing).

    python scripts/analyze_engine_games.py [--since 2026-06-01]

Reports, per completed game: engine side, final margin, whether the human ever
led/scored, when the first capture landed; plus opening ring centroid/spread and
marker footprint. Aggregate line gives the engine's W-L-D and mean margin.
"""
import argparse
import statistics as st
from collections import defaultdict

import lib_game_logs as L


def analyze(sid, rows):
    eng = L.engine_side(rows)
    if eng not in ("WHITE", "BLACK"):
        return None
    hum = "BLACK" if eng == "WHITE" else "WHITE"
    em, hm = L.marker_for(eng), L.marker_for(hum)

    plies = []
    for r in rows:
        npos = r.get("new_position") or {}
        try:
            gs = L.build_state(npos)
        except Exception:
            continue
        _, e_max = L.runs_by_length(gs.board, em)
        _, h_max = L.runs_by_length(gs.board, hm)
        sc = npos.get("scores") or {}
        plies.append({
            "i": len(plies) + 1,
            "e_score": sc.get(eng, 0), "h_score": sc.get(hum, 0),
            "e_max": e_max, "h_max": h_max,
            "e_mk": sum(1 for p in gs.board.pieces.values() if p == em),
            "h_mk": sum(1 for p in gs.board.pieces.values() if p == hm),
        })
    if not plies:
        return None
    final = plies[-1]

    def centroid(side):
        pts = [(ord(s[0]) - 65, int(s[1:]))
               for r in rows
               if (ap := r.get("applied_move") or {}).get("type") == "PLACE_RING"
               and r["pre_position"]["side_to_move"] == side
               and (s := ap.get("source"))]
        if not pts:
            return "?", 0.0
        cx, cy = st.mean(p[0] for p in pts), st.mean(p[1] for p in pts)
        spread = st.mean(((p[0]-cx)**2 + (p[1]-cy)**2) ** 0.5 for p in pts) if len(pts) > 1 else 0.0
        return f"{chr(int(round(cx))+65)}{int(round(cy))}", spread

    e_cent, e_spread = centroid(eng)
    h_cent, h_spread = centroid(hum)
    return {
        "sid": sid[:8], "eng": eng,
        "final": (final["e_score"], final["h_score"]),
        "plies": len(plies),
        "human_led": any(p["h_score"] > p["e_score"] for p in plies),
        "human_scored": final["h_score"] > 0,
        "first_e_cap": next((p["i"] for p in plies if p["e_score"] >= 1), None),
        "first_h_cap": next((p["i"] for p in plies if p["h_score"] >= 1), None),
        "e_cent": e_cent, "e_spread": e_spread,
        "h_cent": h_cent, "h_spread": h_spread,
        "max_e_mk": max(p["e_mk"] for p in plies),
        "max_h_mk": max(p["h_mk"] for p in plies),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None)
    ap.add_argument("--log-dir", default=L.DEFAULT_LOG_DIR)
    args = ap.parse_args()

    games = L.completed_only(L.load_sessions(args.log_dir, args.since))
    res = sorted((a for sid, rows in games.items() if (a := analyze(sid, rows))),
                 key=lambda a: a["sid"])
    if not res:
        print("No completed games found.")
        return

    print("=" * 88 + "\nDOMINANCE\n" + "=" * 88)
    print(f"{'game':<9}{'eng':<6}{'score':<7}{'plies':>6}{'1stEcap':>9}{'1stHcap':>9}"
          f"{'H led?':>8}{'H scored?':>10}")
    for a in res:
        print(f"{a['sid']:<9}{a['eng']:<6}{a['final'][0]}-{a['final'][1]:<5}{a['plies']:>6}"
              f"{str(a['first_e_cap']):>9}{str(a['first_h_cap']):>9}"
              f"{str(a['human_led']):>8}{str(a['human_scored']):>10}")
    margins = [a['final'][0] - a['final'][1] for a in res]
    wins = sum(1 for a in res if a['final'][0] > a['final'][1])
    losses = sum(1 for a in res if a['final'][0] < a['final'][1])
    draws = len(res) - wins - losses
    print(f"\n  engine record: {wins}W-{losses}L-{draws}D   mean margin: {st.mean(margins):+.2f}"
          f"   shutouts: {sum(1 for a in res if a['final'][1] == 0)}/{len(res)}"
          f"   games H ever led: {sum(1 for a in res if a['human_led'])}/{len(res)}")

    print("\n" + "=" * 88 + "\nPLAYSTYLE  (ring opening + marker footprint)\n" + "=" * 88)
    print(f"{'game':<9}{'E open':>9}{'E sprd':>8}{'H open':>9}{'H sprd':>8}{'maxEmk':>8}{'maxHmk':>8}")
    for a in res:
        print(f"{a['sid']:<9}{a['e_cent']:>9}{a['e_spread']:>8.2f}{a['h_cent']:>9}"
              f"{a['h_spread']:>8.2f}{a['max_e_mk']:>8}{a['max_h_mk']:>8}")
    print(f"\n  engine opening spread (mean): {st.mean(a['e_spread'] for a in res):.2f}   "
          f"human: {st.mean(a['h_spread'] for a in res):.2f}   "
          f"(lower = more compact/central)")


if __name__ == "__main__":
    main()
