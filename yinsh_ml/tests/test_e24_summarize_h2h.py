"""Torch-free coverage for the E24 H2H slope summarizer."""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "e24_summarize_h2h",
    Path(__file__).resolve().parents[2] / "scripts" / "e24_summarize_h2h.py",
)
sm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sm)


def _write_pair(d, arm, it, white_wins_w, white_wins_b, black_wins_w, black_wins_b, draw=0):
    # as_white: challenger is white; as_black: challenger is black
    (d / f"{arm}_iter{it}_as_white.json").write_text(json.dumps(
        {"config": {"white_label": arm, "black_label": "frozen_iter1"},
         "wins": {"white": white_wins_w, "black": white_wins_b, "draw": draw}}))
    (d / f"{arm}_iter{it}_as_black.json").write_text(json.dumps(
        {"config": {"white_label": "frozen_iter1", "black_label": arm},
         "wins": {"white": black_wins_w, "black": black_wins_b, "draw": draw}}))


def test_challenger_tally_color_balancing():
    dw = {"wins": {"white": 3, "black": 1, "draw": 0}}  # challenger (white) won 3
    db = {"wins": {"white": 1, "black": 3, "draw": 0}}  # challenger (black) won 3
    ck, fr, dr, tot = sm.challenger_tally(dw, db)
    assert (ck, fr, dr, tot) == (6, 2, 0, 8)


def test_wilson_brackets_half_at_5050():
    lo, hi = sm.wilson(50, 100)
    assert lo < 0.5 < hi


def test_collect_and_slope_picks_up_lr_tags(tmp_path):
    # iter0: challenger 6-2 (score .75); iter1: 4-4 (score .50) -> slope -0.25/iter
    _write_pair(tmp_path, "lr1e-4", 0, 3, 1, 1, 3)
    _write_pair(tmp_path, "lr1e-4", 1, 2, 2, 2, 2)
    data = sm.collect(str(tmp_path))
    assert set(data) == {"lr1e-4"}              # hyphenated lr tag parsed (e19's regex would miss this)
    assert sorted(data["lr1e-4"]) == [0, 1]
    s0 = sm.challenger_tally(data["lr1e-4"][0]["white"], data["lr1e-4"][0]["black"])
    s1 = sm.challenger_tally(data["lr1e-4"][1]["white"], data["lr1e-4"][1]["black"])
    score0 = (s0[0] + 0.5 * s0[2]) / s0[3]
    score1 = (s1[0] + 0.5 * s1[2]) / s1[3]
    assert score0 == 0.75 and score1 == 0.5
    assert abs(sm.linfit_slope([0, 1], [score0, score1]) - (-0.25)) < 1e-9
    # summary renders without error and names the arm
    assert "lr1e-4" in sm.summarize(data)
