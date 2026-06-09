#!/usr/bin/env python3
"""How the engine handles human scoring threats, with move-generator-grade
completability (not the geometric open-cell proxy).

    python scripts/threat_response.py [--since 2026-06-01]

For every human-to-move MAIN_GAME position it asks the real question — does the
human have a legal move that completes a 5-row right now? (``can_complete`` via
simulation) — then tracks whether the human took it. It also audits the old
geometric "live 4-run" proxy against that ground truth, and measures flip-profit:
when the engine breaks a human 4-run, does it gain markers on the same move
(defense doubling as offense)?
"""
import argparse
import statistics as st

import lib_game_logs as L


def analyze(sid, rows):
    eng = L.engine_side(rows)
    if eng not in ("WHITE", "BLACK"):
        return None
    hum = "BLACK" if eng == "WHITE" else "WHITE"
    em, hm = L.marker_for(eng), L.marker_for(hum)

    trace = []
    for r in rows:
        npos = r.get("new_position") or {}
        try:
            gs = L.build_state(npos)
        except Exception:
            continue
        h4_all, h4_live = L.four_runs_with_liveness(gs.board, hm)
        stm = npos.get("side_to_move")
        phase = npos.get("phase")
        ict = None  # immediate completion threat: human can complete THIS move
        if stm == hum and phase == "MAIN_GAME":
            ict = L.can_complete(gs, hum)
        sc = npos.get("scores") or {}
        trace.append({
            "mover": "E" if r["pre_position"]["side_to_move"] == eng else "H",
            "h4_all": h4_all, "h4_live": h4_live, "ict": ict,
            "e_mk": sum(1 for p in gs.board.pieces.values() if p == em),
            "e_score": sc.get(eng, 0), "h_score": sc.get(hum, 0),
        })
    if not trace:
        return None

    # Real completion chances + conversion
    ict_total = ict_taken = 0
    for i, t in enumerate(trace):
        if not t["ict"]:
            continue
        ict_total += 1
        base = t["h_score"]
        took = False
        for j in range(i + 1, len(trace)):
            if trace[j]["mover"] != "H":
                break
            if trace[j]["h_score"] > base:
                took = True
                break
        ict_taken += took

    # Proxy audit: among human-to-move positions, how often "live 4-run" present
    # actually corresponds to a real completion chance.
    hum_to_move = [t for t in trace if t["ict"] is not None]
    proxy_flagged = [t for t in hum_to_move if t["h4_live"]]
    proxy_tp = sum(1 for t in proxy_flagged if t["ict"])
    ict_positions = [t for t in hum_to_move if t["ict"]]
    proxy_caught = sum(1 for t in ict_positions if t["h4_live"])

    # Flip-profit on engine engagements
    marker_gain = []
    live_faced = live_neutralized = 0
    for k in range(1, len(trace)):
        cur, prev = trace[k], trace[k - 1]
        if cur["mover"] != "E" or not prev["h4_live"]:
            continue
        live_faced += len(prev["h4_live"])
        neutralized = prev["h4_live"] - cur["h4_all"]
        if neutralized:
            live_neutralized += len(neutralized)
            marker_gain.append(cur["e_mk"] - prev["e_mk"])

    return {
        "sid": sid[:8], "eng": eng, "score": (trace[-1]["e_score"], trace[-1]["h_score"]),
        "ict_total": ict_total, "ict_taken": ict_taken,
        "proxy_flagged": len(proxy_flagged), "proxy_tp": proxy_tp,
        "ict_positions": len(ict_positions), "proxy_caught": proxy_caught,
        "live_faced": live_faced, "live_neutralized": live_neutralized,
        "marker_gain": marker_gain,
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

    print("=" * 92)
    print("THREAT RESPONSE  (move-generator completability)")
    print("=" * 92)
    print(f"{'game':<9}{'eng':<6}{'Hpts':>5}{'realThreats':>12}{'taken':>7}{'declined':>9}"
          f"{'flipNeutral':>12}{'avgMkGain':>10}")
    TICT = TTAKEN = 0
    allmk = []
    pf = ptp = ictp = pcaught = 0
    for a in res:
        TICT += a["ict_total"]; TTAKEN += a["ict_taken"]
        allmk += a["marker_gain"]
        pf += a["proxy_flagged"]; ptp += a["proxy_tp"]
        ictp += a["ict_positions"]; pcaught += a["proxy_caught"]
        mg = f"{st.mean(a['marker_gain']):+.1f}" if a["marker_gain"] else "—"
        print(f"{a['sid']:<9}{a['eng']:<6}{a['score'][1]:>5}{a['ict_total']:>12}"
              f"{a['ict_taken']:>7}{a['ict_total']-a['ict_taken']:>9}"
              f"{a['live_neutralized']:>12}{mg:>10}")

    print("-" * 92)
    conv = f"{TTAKEN}/{TICT} ({TTAKEN/TICT*100:.0f}%)" if TICT else "—"
    print(f"\nReal immediate completion chances the human got: {TICT}   taken: {conv}")
    if allmk:
        gained = sum(1 for x in allmk if x > 0)
        print(f"Flip-profit when neutralizing a human 4-run: avg marker delta "
              f"{st.mean(allmk):+.2f} (median {st.median(allmk):+.0f}); engine gained on "
              f"{gained}/{len(allmk)} ({gained/len(allmk)*100:.0f}%). ~+1 is the move's own "
              f"marker drop; the rest is flipped human markers.")
    print(f"\nGeometric proxy audit (old 'live 4-run' check):")
    if pf:
        print(f"  positions it flagged: {pf}, of which a real completion existed: "
              f"{ptp} -> precision {ptp/pf*100:.0f}%  (it overcounted threats by "
              f"{(pf-ptp)/pf*100:.0f}%)")
    if ictp:
        print(f"  real completion positions: {ictp}, of which the proxy caught: "
              f"{pcaught} -> recall {pcaught/ictp*100:.0f}%  (misses flip/merge completions "
              f"with no standing 4-run)")


if __name__ == "__main__":
    main()
