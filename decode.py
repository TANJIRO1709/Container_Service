"""Phase 5: probability calibration + per-entity expected-F0.5 decoding.

Why a single threshold is the wrong tool
----------------------------------------
The metric is macro-averaged PER ENTITY, and for a prediction set P against a truth
set G:   F0.5 = 1.25*TP / (|P| + 0.25*|G|),  with an empty P scoring 1.0 when G is
empty and 0.0 otherwise. So the value of adding one more candidate depends on how many
you have already taken and on how likely the entity is to be a singleton — both of
which vary per entity. One global cut-off cannot express that.

The decoder
-----------
For an entity whose calibrated candidate probabilities are p_1 >= p_2 >= ... >= p_n,
taking the top k has expected score approximately

    k >= 1 :  E[F] ~= 1.25 * (sum of top-k p) / (k + 0.25 * sum of all p)
    k  = 0 :  E[F]  = prod(1 - p_i)          (the entity is truly a singleton)

Taking the best k per entity is optimal under the plug-in approximation, and it
abstains automatically when the evidence is weak — exactly what a precision-heavy
metric with singleton credit rewards. Two tuned knobs correct for the approximation:
  `alpha`  scales the expected size of G (raise it to be more conservative)
  `empty_boost` scales the empty-set score (raise it to predict more singletons)
  `floor`  drops candidates below this probability before decoding — the plug-in
           expectation is optimistic about long tails of weak candidates, and a floor
           is worth more than either other knob in testing

Both are fitted on held-out validation entities, never on the training pairs.
"""
import numpy as np
import polars as pl

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:                                    # pragma: no cover
    IsotonicRegression = None


# --------------------------------------------------------------------------- calibration
def fit_calibrator(prob, label, n_bins=2000):
    """Isotonic calibration: raw model scores -> probabilities that mean what they say.

    The decoder multiplies and compares probabilities, so a model that is merely well
    RANKED is not enough; the numbers themselves must be right.
    """
    if IsotonicRegression is None or len(prob) < 100:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    # bin first: isotonic on tens of millions of points is slow and adds nothing
    order = np.argsort(prob)
    p, y = np.asarray(prob)[order], np.asarray(label)[order]
    edges = np.linspace(0, len(p), min(n_bins, len(p)) + 1).astype(int)
    xs = np.array([p[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    ys = np.array([y[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    ws = np.array([b - a for a, b in zip(edges[:-1], edges[1:]) if b > a], dtype=float)
    iso.fit(xs, ys, sample_weight=ws)
    return iso


def apply_calibrator(iso, prob):
    if iso is None:
        return np.asarray(prob, dtype=np.float64)
    return np.clip(iso.predict(np.asarray(prob, dtype=np.float64)), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- decoder
def decode(df: pl.DataFrame, prob_col: str = "p_cal", alpha: float = 1.0,
           empty_boost: float = 1.0, max_k: int = 12, floor: float = 0.0) -> pl.DataFrame:
    """Choose, per entity, how many top candidates to keep. Adds a boolean `keep`.

    df must contain `idx` (entity), `idx2` (candidate) and `prob_col`.
    """
    if df.height == 0:
        return df.with_columns(pl.lit(False).alias("keep"))
    d = df.filter(pl.col(prob_col) >= floor) if floor > 0 else df
    if d.height == 0:
        return df.with_columns(pl.lit(False).alias("keep"))
    d = d.sort(["idx", prob_col], descending=[False, True]).with_columns(
        (pl.int_range(pl.len()).over("idx") + 1).alias("k"),
        pl.col(prob_col).cum_sum().over("idx").alias("csum"),
        pl.col(prob_col).sum().over("idx").alias("psum"),
        (1.0 - pl.col(prob_col)).log().sum().over("idx").alias("log_empty"),
    )
    d = d.with_columns(
        (1.25 * pl.col("csum") / (pl.col("k") + alpha * 0.25 * pl.col("psum"))).alias("ef"),
        (pl.col("log_empty").exp() * empty_boost).alias("ef0"),
    )
    # best k per entity, capped: beyond max_k the denominator dominates anyway
    best = d.filter(pl.col("k") <= max_k).group_by("idx").agg(
        pl.col("k").sort_by("ef", descending=True).first().alias("best_k"),
        pl.col("ef").max().alias("best_ef"),
        pl.col("ef0").first().alias("ef0"),
    ).with_columns(
        pl.when(pl.col("ef0") >= pl.col("best_ef")).then(0).otherwise(pl.col("best_k")).alias("take")
    )
    keep = d.join(best.select("idx", "take"), on="idx", how="left").with_columns(
        (pl.col("k") <= pl.col("take").fill_null(0)).alias("keep")
    ).filter("keep").select("idx", "idx2")
    return df.join(keep.with_columns(pl.lit(True).alias("keep")), on=["idx", "idx2"], how="left") \
             .with_columns(pl.col("keep").fill_null(False))


def tune(val: pl.DataFrame, ent: pl.DataFrame, score_fn, prob_col: str = "p_cal",
         alphas=(0.8, 1.0, 1.3, 1.7, 2.2), boosts=(0.8, 1.0, 1.3, 1.8, 2.5),
         floors=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Grid-search the decoder knobs on held-out entities. Returns (alpha, boost, floor, report)."""
    best = (1.0, 1.0, 0.0, {"macro_f05": -1.0})
    for fl in floors:
        for al in alphas:
            for eb in boosts:
                r = score_fn(ent, decode(val, prob_col, al, eb, floor=fl), "keep")
                if r["macro_f05"] > best[3]["macro_f05"]:
                    best = (al, eb, fl, r)
    return best
