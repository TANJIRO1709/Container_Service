"""Phase 2 training: typo-aware budgeted blocking + LightGBM, threshold tuned for macro F0.5.

Usage (from student_resource/):
  python3 src/phase2_train.py --data dataset --models models
Split of sampled S1 entities: 70% train / 10% early-stopping / 20% validation (threshold + report).
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import polars as pl

from evaluate import macro_f05_frame, tune_threshold
from io_utils import explode_ids, read_tsv
from student_resource.src.pipeline2 import FEATURES, Corpus2, X_of


def get_model(n_estimators):
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=127, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, verbose=-1)
    except (ImportError, OSError) as e:   # e.g. missing libomp on macOS
        print(f"!! LightGBM unavailable ({e}); falling back to sklearn HistGradientBoosting")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hgb", HistGradientBoostingClassifier(
            max_iter=n_estimators, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=100,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)


def main(a):
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    d = f"{a.data}/train"
    s1 = read_tsv(f"{d}/train_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/train_source2.tsv"), read_tsv(f"{d}/train_source3.tsv")])
    gt = read_tsv(f"{d}/train_ground_truth.tsv")
    if a.s1_sample and a.s1_sample < s1.height:
        s1 = s1.sample(a.s1_sample, seed=a.seed)
    log(f"S1={s1.height:,}  targets={tgt.height:,}")

    corpus = Corpus2(s1, tgt, max_df=a.max_df, budget=a.budget, spell=not a.no_spell)
    log("index built")

    id1 = corpus.s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
    id2 = corpus.tg.select(pl.col("entity_id").alias("target_id"), pl.col("idx").alias("idx2"))
    gp = explode_ids(gt, "matched_entity_ids").join(id1, on="source1_entity_id").join(id2, on="target_id") \
        .select("idx", "idx2", pl.lit(1, pl.Int8).alias("label"))
    ent = id1.join(gp.group_by("idx").len("G"), on="idx", how="left").select("idx", pl.col("G").fill_null(0))

    parts = []
    for i, f in enumerate(corpus.iter_chunks(a.K, a.K_pre, a.chunk, a.cost_cap)):
        parts.append(f.join(gp, on=["idx", "idx2"], how="left").with_columns(pl.col("label").fill_null(0)))
        if i % 5 == 0:
            log(f"  chunk {i}: {f.height:,} pairs")
    rows = pl.concat(parts)
    del parts

    rc = rows["label"].sum() / max(gp.height, 1)
    log(f"candidates/S1={rows.height / corpus.s1.height:.1f}  BLOCKING RECALL CEILING={rc:.4f}")
    # recall ceiling as a function of K (to choose K without re-running)
    pos_rank = rows.filter(pl.col("label") == 1)["rank"]
    print("  recall@K:", {k: round(float((pos_rank < k).sum() / gp.height), 4) for k in (5, 10, 20, 30, 50, 75, 100) if k <= a.K})

    rng = np.random.default_rng(a.seed)
    u = rng.random(corpus.s1.height)
    split = pl.Series("idx", np.arange(corpus.s1.height, dtype=np.uint32))
    tr_ids, es_ids, va_ids = split.filter(u < 0.7), split.filter((u >= 0.7) & (u < 0.8)), split.filter(u >= 0.8)
    inn = lambda s: pl.col("idx").is_in(s.implode())
    tr, es, va = rows.filter(inn(tr_ids)), rows.filter(inn(es_ids)), rows.filter(inn(va_ids))
    ent_va = ent.filter(inn(va_ids))
    del rows

    oracle = macro_f05_frame(ent_va, va.with_columns((pl.col("label") == 1).alias("p")), "p")
    log(f"ORACLE macro F0.5 = {oracle['macro_f05']:.4f}")

    kind, model = get_model(a.n_estimators)
    log(f"training {kind} on {tr.height:,} pairs ({int(tr['label'].sum()):,} positives)")
    Xtr, ytr = X_of(tr), tr["label"].to_numpy()
    if kind == "lightgbm":
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(X_of(es), es["label"].to_numpy())], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        log(f"best iteration: {model.best_iteration_}")
    else:
        model.fit(Xtr, ytr)
    del Xtr, ytr, tr

    va = va.with_columns(pl.Series("prob", model.predict_proba(X_of(va))[:, 1]))
    t, r = tune_threshold(ent_va, va)
    log(f"VALIDATION  threshold={t}  macro F0.5={r['macro_f05']:.4f}  singleton_acc={r['singleton_acc']:.4f}  "
        f"matched-entity F0.5={r['matched_f05']:.4f}   (oracle {oracle['macro_f05']:.4f})")

    if kind == "lightgbm":
        imp = sorted(zip(FEATURES, model.booster_.feature_importance("gain")), key=lambda x: -x[1])
        tot = sum(v for _, v in imp)
        print("  top features (gain share):", ", ".join(f"{k}={v / tot:.3f}" for k, v in imp[:15]))

    os.makedirs(a.models, exist_ok=True)
    cfg = {k: getattr(a, k) for k in ("K", "K_pre", "max_df", "budget", "no_spell")}
    joblib.dump({"model": model, "kind": kind, "threshold": t, "features": FEATURES, **cfg}, f"{a.models}/phase2.joblib")
    va.select("idx", "idx2", "label", "prob").write_parquet(f"{a.models}/phase2_val_preds.parquet")
    ent_va.write_parquet(f"{a.models}/phase2_val_entities.parquet")
    with open(f"{a.models}/phase2_report.json", "w") as fh:
        json.dump({"recall_ceiling": rc, "oracle": oracle, "val": r, "threshold": t, "model": kind, "args": vars(a)}, fh, indent=2)
    log(f"saved -> {a.models}/phase2.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--s1-sample", type=int, default=250_000, help="S1 entities used (0 = all)")
    ap.add_argument("--K", type=int, default=50, help="final candidates per S1")
    ap.add_argument("--K-pre", dest="K_pre", type=int, default=150, help="stage-1 candidates per S1")
    ap.add_argument("--max-df", type=int, default=10_000, help="index keys up to this posting length")
    ap.add_argument("--budget", type=int, default=3_000, help="per-record retrieval budget (sum of posting lengths)")
    ap.add_argument("--no-spell", action="store_true", help="disable typo correction (ablation)")
    ap.add_argument("--cost-cap", type=int, default=20_000_000, help="max retrieval rows per chunk (lower = less RAM)")
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
