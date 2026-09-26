"""Phase 1 training: blocking on train, logistic-regression matcher, threshold
tuned for macro per-entity F0.5 on a held-out S1 split.

Usage (from student_resource/):
  python src/phase1_train.py --data dataset --models models
Reports: blocking recall ceiling, oracle F0.5 (perfect matcher on our candidates),
validation macro F0.5 at the tuned threshold.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from io_utils import explode_ids, read_tsv
from pipeline import FEATURES, Corpus, X_of


def macro_f05_frame(ent: pl.DataFrame, rows: pl.DataFrame, pred_col: str) -> dict:
    """ent: (idx, G) for ALL eval entities; rows: candidate rows with idx,label,pred_col(bool)."""
    agg = rows.group_by("idx").agg(
        (pl.col(pred_col) & (pl.col("label") == 1)).sum().alias("tp"),
        pl.col(pred_col).sum().alias("np"),
    )
    e = ent.join(agg, on="idx", how="left").fill_null(0)
    f = e.select(
        pl.when((pl.col("G") == 0) & (pl.col("np") == 0)).then(1.0)
        .when((pl.col("G") == 0) | (pl.col("np") == 0)).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (pl.col("np") + 0.25 * pl.col("G")))
        .alias("f"),
        (pl.col("G") == 0).alias("single"),
    )
    return {
        "macro_f05": f["f"].mean(),
        "singleton_acc": f.filter("single")["f"].mean() if f["single"].any() else float("nan"),
        "matched_f05": f.filter(~pl.col("single"))["f"].mean(),
    }


def main(a):
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    d = f"{a.data}/train"
    s1 = read_tsv(f"{d}/train_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/train_source2.tsv"), read_tsv(f"{d}/train_source3.tsv")])
    gt = read_tsv(f"{d}/train_ground_truth.tsv")
    if a.s1_sample and a.s1_sample < s1.height:
        s1 = s1.sample(a.s1_sample, seed=a.seed)   # S1 sampled; target pool stays FULL (realistic competition)
    print(f"S1={s1.height:,}  targets={tgt.height:,}  [{time.time() - t0:.0f}s]")

    corpus = Corpus(s1, tgt, max_df=a.max_df)
    print(f"index built [{time.time() - t0:.0f}s]")

    # ground-truth pairs in idx space
    id1 = corpus.s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
    id2 = corpus.tg.select(pl.col("entity_id").alias("target_id"), pl.col("idx").alias("idx2"))
    gp = explode_ids(gt, "matched_entity_ids").join(id1, on="source1_entity_id").join(id2, on="target_id") \
        .select("idx", "idx2", pl.lit(1, pl.Int8).alias("label"))
    ent = id1.join(gp.group_by("idx").len("G"), on="idx", how="left").select("idx", pl.col("G").fill_null(0))

    parts = []
    for i, f in enumerate(corpus.iter_chunks(a.K, a.chunk)):
        parts.append(f.join(gp, on=["idx", "idx2"], how="left").with_columns(pl.col("label").fill_null(0)))
        print(f"  chunk {i}: {f.height:,} pairs [{time.time() - t0:.0f}s]")
    rows = pl.concat(parts)

    recall_ceiling = rows["label"].sum() / max(gp.height, 1)
    print(f"\ncandidates/S1: {rows.height / corpus.s1.height:.1f}   positives rate: {rows['label'].mean():.4f}")
    print(f"BLOCKING RECALL CEILING: {recall_ceiling:.4f}")

    # split by S1 entity (group split)
    val_mask = rng.random(corpus.s1.height) < a.val_frac
    val_idx = pl.Series("idx", np.nonzero(val_mask)[0].astype(np.uint32))
    is_val = pl.col("idx").is_in(val_idx.implode())
    tr, va = rows.filter(~is_val), rows.filter(is_val)
    ent_va = ent.filter(is_val)

    oracle = macro_f05_frame(ent_va, va.with_columns((pl.col("label") == 1).alias("p")), "p")
    print(f"ORACLE macro F0.5 (perfect matcher on these candidates): {oracle['macro_f05']:.4f}")

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    model.fit(X_of(tr), tr["label"].to_numpy())
    va = va.with_columns(pl.Series("prob", model.predict_proba(X_of(va))[:, 1]))

    best = (0.5, {"macro_f05": -1})
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        r = macro_f05_frame(ent_va, va.with_columns((pl.col("prob") >= t).alias("p")), "p")
        if r["macro_f05"] > best[1]["macro_f05"]:
            best = (float(t), r)
    t, r = best
    print(f"\nVALIDATION  threshold={t}  macro F0.5={r['macro_f05']:.4f}  "
          f"singleton_acc={r['singleton_acc']:.4f}  matched-entity F0.5={r['matched_f05']:.4f}")
    empty = macro_f05_frame(ent_va, va.with_columns(pl.lit(False).alias("p")), "p")["macro_f05"]
    print(f"(all-empty baseline on val = {empty:.4f})")

    coefs = dict(zip(FEATURES, np.round(model[-1].coef_[0], 3).tolist()))
    print("LR coefficients (standardized):", json.dumps(coefs))

    os.makedirs(a.models, exist_ok=True)
    joblib.dump({"model": model, "threshold": t, "features": FEATURES, "K": a.K, "max_df": a.max_df},
                f"{a.models}/phase1.joblib")
    with open(f"{a.models}/phase1_report.json", "w") as fh:
        json.dump({"recall_ceiling": recall_ceiling, "oracle": oracle, "val": r, "threshold": t,
                   "all_empty_val": empty, "args": vars(a)}, fh, indent=2)
    print(f"saved -> {a.models}/phase1.joblib  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--s1-sample", type=int, default=300_000, help="S1 entities used for training (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--K", type=int, default=30, help="candidates kept per S1")
    ap.add_argument("--max-df", type=int, default=500, help="drop blocking keys more common than this")
    ap.add_argument("--chunk", type=int, default=50_000, help="S1 entities per processing chunk")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
