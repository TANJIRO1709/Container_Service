"""Shared evaluation helpers (vectorised macro per-entity F0.5)."""
import numpy as np
import polars as pl


def macro_f05_frame(ent: pl.DataFrame, rows: pl.DataFrame, pred_col: str) -> dict:
    """ent: (idx, G) for ALL evaluated S1 entities (G = true match count, incl. ones blocking missed).
    rows: candidate rows with idx, label (0/1), pred_col (bool)."""
    agg = rows.group_by("idx").agg(
        (pl.col(pred_col) & (pl.col("label") == 1)).sum().alias("tp"),
        pl.col(pred_col).sum().alias("np"),
    )
    e = ent.join(agg, on="idx", how="left").fill_null(0)
    f = e.select(
        pl.when((pl.col("G") == 0) & (pl.col("np") == 0)).then(1.0)
        .when((pl.col("G") == 0) | (pl.col("np") == 0)).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (pl.col("np") + 0.25 * pl.col("G"))).alias("f"),
        (pl.col("G") == 0).alias("single"),
    )
    sing = f.filter("single")
    return {"macro_f05": float(f["f"].mean()),
            "singleton_acc": float(sing["f"].mean()) if sing.height else float("nan"),
            "matched_f05": float(f.filter(~pl.col("single"))["f"].mean())}


def tune_threshold(ent, rows, prob_col="prob", grid=None):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 3) if grid is None else grid
    best_t, best = 0.5, {"macro_f05": -1.0}
    for t in grid:
        r = macro_f05_frame(ent, rows.with_columns((pl.col(prob_col) >= t).alias("p")), "p")
        if r["macro_f05"] > best["macro_f05"]:
            best_t, best = float(t), r
    return best_t, best
