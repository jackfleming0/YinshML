"""Re-fit heuristic feature weights from game outcomes (B).

This is the *math core* for re-fitting the production heuristic's per-phase
feature weights — pure NumPy, no torch / sklearn / IO — so it is unit-testable
and importable anywhere. CLI + data loading live in
``scripts/experiments/fit_heuristic_weights.py``.

Why re-fit: reviewing a strong human game showed ``potential_runs_count`` was
silently 0 everywhere (now fixed) and that the weight (0.171) it carried was fit
against that constant. Re-fitting on real outcomes — with the feature now live —
is the corrective step. See docs/game_reviews/bga_862307561_review.md.

Conventions, matched to the production evaluator + WeightManager:
- Phases are ``early``/``mid``/``late`` (split by move count).
- Features are all *differential* (mine - opponent) and oriented so higher is
  better for the perspective player; weights are therefore **non-negative** and
  WeightManager clamps them to [0, 50]. A fitted coefficient that comes out
  negative means "given the others, this signal points the wrong way" — we clamp
  it to 0 (i.e. drop it) rather than invert the feature.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The 6 weighted production features, in a fixed order (matches
# WeightManager.VALID_FEATURES and evaluator._feature_names).
PRODUCTION_FEATURES: Tuple[str, ...] = (
    "completed_runs_differential",
    "potential_runs_count",
    "connected_marker_chains",
    "ring_positioning",
    "ring_spread",
    "board_control",
)

PHASES: Tuple[str, ...] = ("early", "mid", "late")

WEIGHT_MIN, WEIGHT_MAX = 0.0, 50.0


def phase_of_move_count(move_count: int, early_max: int, mid_max: int) -> str:
    """Bucket a position into early/mid/late by move count (evaluator's rule)."""
    if move_count <= early_max:
        return "early"
    if move_count <= mid_max:
        return "mid"
    return "late"


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)  # leave constant columns untouched
    return (X - mu) / sd, mu, sd


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    iters: int = 500,
    lr: float = 0.5,
) -> np.ndarray:
    """Logistic regression coefficients (one per column of X) via gradient
    descent on standardized features, de-standardized back to raw scale.

    The intercept is fit but not returned — only the per-feature slopes matter
    for the heuristic's linear combination. Returns coefficients on the raw
    (un-standardized) feature scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    Xs, mu, sd = _standardize(X)
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        grad_w = Xs.T @ err / n + l2 * w / n
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    # de-standardize slopes: coef_raw = coef_std / sd
    return w / sd


def fit_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature Pearson correlation with the outcome (the original
    methodology). Constant columns yield 0."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    yc = y - y.mean()
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        xj = X[:, j] - X[:, j].mean()
        denom = np.sqrt((xj * xj).sum() * (yc * yc).sum())
        out[j] = (xj * yc).sum() / denom if denom > 1e-12 else 0.0
    return out


def clamp_and_scale(coefs: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale coefficients then clamp to the WeightManager range [0, 50].

    Negative coefficients clamp to 0 (the feature is dropped). ``scale`` maps
    the fitter's natural coefficient magnitude onto the heuristic's expected
    weight magnitude (the production defaults are O(1-12)).
    """
    return np.clip(coefs * scale, WEIGHT_MIN, WEIGHT_MAX)


Sample = Tuple[str, Dict[str, float], int]  # (phase, feature_dict, outcome_label)


def fit_weights_from_samples(
    samples: Sequence[Sample],
    *,
    method: str = "logreg",
    features: Sequence[str] = PRODUCTION_FEATURES,
    scale: float = 10.0,
    l2: float = 1.0,
    min_samples_per_phase: int = 50,
    fallback: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit a full per-phase weight config from labeled position samples.

    Args:
        samples: ``(phase, {feature: value}, label)`` triples. ``label`` is 1 if
            the perspective player ultimately won, else 0.
        method: ``"logreg"`` or ``"correlation"``.
        features: feature order to fit (defaults to the 6 production features).
        scale: multiplier applied before clamping to [0, 50].
        l2: ridge strength for logreg.
        min_samples_per_phase: phases with fewer samples fall back (see below).
        fallback: per-phase weights to use for under-sampled phases. If None,
            such phases get all-zero weights (caller should treat as a warning).

    Returns:
        ``{phase: {feature: weight}}`` with every PHASE and every requested
        feature present, weights clamped to [0, 50] — i.e. directly loadable by
        WeightManager.
    """
    if method not in ("logreg", "correlation"):
        raise ValueError(f"unknown method {method!r}")

    feats = list(features)
    by_phase: Dict[str, List[Sample]] = {p: [] for p in PHASES}
    for phase, fd, label in samples:
        if phase in by_phase:
            by_phase[phase].append((phase, fd, label))

    result: Dict[str, Dict[str, float]] = {}
    for phase in PHASES:
        rows = by_phase[phase]
        if len(rows) < min_samples_per_phase:
            result[phase] = dict(
                (fallback or {}).get(phase, {f: 0.0 for f in feats})
            )
            # ensure all features present
            for f in feats:
                result[phase].setdefault(f, 0.0)
            continue
        X = np.array([[fd.get(f, 0.0) for f in feats] for _p, fd, _l in rows], dtype=float)
        y = np.array([lab for _p, _fd, lab in rows], dtype=float)
        coefs = fit_logistic(X, y, l2=l2) if method == "logreg" else fit_correlation(X, y)
        w = clamp_and_scale(coefs, scale=scale)
        result[phase] = {f: float(w[i]) for i, f in enumerate(feats)}
    return result
