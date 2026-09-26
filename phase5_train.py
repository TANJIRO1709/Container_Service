"""Phase 5 training: Phase-4 pipeline + isotonic calibration + per-entity
expected-F0.5 decoding (decode.py), which replaces the single global threshold.

Pipeline:
  pass 1  blocking -> base features -> BASE model (out-of-fold probabilities)
  derive  per-target competition stats + per-entity anchor support
  pass 2  FINAL model on base features + cross-record features
Both models are trained here; the threshold is tuned on a held-out 20% of S1 entities.

Usage (from student_resource/):
  py src\\phase5_train.py --data dataset --models models --batch 250000
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import polars as pl

from competition import COMP_FEATURES, add_competition, add_support, enforce_exclusivity, target_stats
from decode import apply_calibrator, decode, fit_calibrator, tune as tune_decoder
from evaluate import macro_f05_frame, tune_threshold
from io_utils import explode_ids, read_tsv
from pipeline3 import FEATURES, Corpus3, X_of


def get_model(n_estimators, seed=0):
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=127, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1)
    except (ImportError, OSError) as e:
        print(f"!! LightGBM unavailable ({e}); using sklearn HistGradientBoosting")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hgb", HistGradientBoostingClassifier(
            max_iter=n_estimators, learning_rate=0.08, max_leaf_nodes=127, min_samples_leaf=100,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=seed)


def fit(kind, model, Xtr, ytr, Xes=None, yes=None):
    if kind == "lightgbm" and Xes is not None:
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(Xtr, ytr)
    return model


def X2(f):
    return f.select(FEATURES + COMP_FEATURES).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)


def main(a):
    t0 = time.time()
    os.makedirs(a.models, exist_ok=True)
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    d = f"{a.data}/train"
    s1 = read_tsv(f"{d}/train_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/train_source2.tsv"), read_tsv(f"{d}/train_source3.tsv")])
    gt = read_tsv(f"{d}/train_ground_truth.tsv")
    if a.s1_sample and a.s1_sample < s1.height:
        s1 = s1.sample(a.s1_sample, seed=a.seed)
    log(f"S1={s1.height:,}  targets={tgt.height:,}")

    corpus = Corpus3(s1, tgt, max_df=a.max_df, budget=a.budget, spell=not a.no_spell, batch=a.batch)
    log("index built")

    id1 = corpus.s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
    id2 = corpus.tg.select(pl.col("entity_id").alias("target_id"), pl.col("idx").alias("idx2"))
    gp = explode_ids(gt, "matched_entity_ids").join(id1, on="source1_entity_id").join(id2, on="target_id") \
        .select("idx", "idx2", pl.lit(1, pl.Int8).alias("label"))
    ent = id1.join(gp.group_by("idx").len("G"), on="idx", how="left").select("idx", pl.col("G").fill_null(0))

    # is the exclusivity assumption actually true in this ground truth?
    shared = gp.group_by("idx2").agg(pl.col("idx").n_unique().alias("n")).filter(pl.col("n") > 1).height
    log(f"EXCLUSIVITY CHECK: {shared:,} targets matched to >1 S1 of {gp['idx2'].n_unique():,} "
        f"-> {'HOLDS' if shared == 0 else 'VIOLATED'}")

    # ---------------------------------------------------------------- pass 1
    parts = []
    for i, f in enumerate(corpus.iter_chunks(a.K, a.K_pre, a.chunk, a.cost_cap, a.K_b, a.K_pre_b)):
        parts.append(f.join(gp, on=["idx", "idx2"], how="left").with_columns(pl.col("label").fill_null(0)))
        if i % 10 == 0:
            log(f"  chunk {i}: {f.height:,} pairs")
    rows = pl.concat(parts, rechunk=False)
    del parts
    rc = rows["label"].sum() / max(gp.height, 1)
    log(f"candidates/S1={rows.height / corpus.s1.height:.1f}  BLOCKING RECALL CEILING={rc:.4f}")

    rng = np.random.default_rng(a.seed)
    u = rng.random(corpus.s1.height)
    allidx = pl.Series("idx", np.arange(corpus.s1.height, dtype=np.uint32))
    tr_ids, es_ids, va_ids = allidx.filter(u < 0.7), allidx.filter((u >= 0.7) & (u < 0.8)), allidx.filter(u >= 0.8)
    inn = lambda s: pl.col("idx").is_in(s.implode())
    ent_va = ent.filter(inn(va_ids))

    # keep every positive but only a fraction of negatives when FITTING: the full
    # candidate table is far larger than RAM at scale, and boosting is insensitive to
    # dropping easy negatives. Scoring and evaluation still use every row.
    def fit_rows(df):
        if a.neg_sample >= 1.0:
            return df
        return df.filter((pl.col("label") == 1)
                         | (pl.col("idx2").hash(seed=13) % 10000 < int(a.neg_sample * 10000)))

    kind, base = get_model(a.n_estimators, a.seed)
    tr, es = fit_rows(rows.filter(inn(tr_ids))), fit_rows(rows.filter(inn(es_ids)))
    log(f"training BASE {kind} on {tr.height:,} pairs")
    base = fit(kind, base, X_of(tr), tr["label"].to_numpy(), X_of(es), es["label"].to_numpy())
    del tr, es

    # out-of-fold-ish base probabilities for every pair (train rows are in-sample; the
    # threshold and the reported score come from the untouched validation entities)
    rows = rows.with_columns(pl.Series("p_base", base.predict_proba(X_of(rows))[:, 1]))
    va_base = rows.filter(inn(va_ids))
    t_b, r_b = tune_threshold(ent_va, va_base.rename({"p_base": "prob"}))
    log(f"BASE validation macro F0.5={r_b['macro_f05']:.4f} (threshold {t_b})")

    # ---------------------------------------------------------------- cross-record
    tstats = target_stats(rows.select("idx", "idx2", pl.col("p_base").alias("prob")))
    log(f"competition stats for {tstats.height:,} targets")
    # spill per-chunk cross-record rows to parquet, then free `rows`: holding the base
    # table and the enriched table in memory at once is the peak of the whole script
    tmp = os.path.join(a.models, "_p5_tmp")
    os.makedirs(tmp, exist_ok=True)
    for old_f in os.listdir(tmp):
        os.remove(os.path.join(tmp, old_f))
    rows = rows.sort("idx")
    n_part = 0
    for lo in range(0, corpus.s1.height, a.chunk):
        c = rows.filter((pl.col("idx") >= lo) & (pl.col("idx") < lo + a.chunk))
        if not c.height:
            continue
        add_support(add_competition(c, tstats), corpus.tg, a.n_anchor) \
            .select("idx", "idx2", "label", *FEATURES, *COMP_FEATURES) \
            .write_parquet(f"{tmp}/part_{n_part:05d}.parquet")
        n_part += 1
    del rows
    lf = pl.scan_parquet(f"{tmp}/*.parquet")
    log(f"cross-record features built ({n_part} parts)")

    kind2, final = get_model(a.n_estimators, a.seed)
    tr = fit_rows(lf.filter(inn(tr_ids)).collect())
    Xtr, ytr = X2(tr), tr["label"].to_numpy()
    del tr
    es = fit_rows(lf.filter(inn(es_ids)).collect())
    log(f"training FINAL {kind2} on {len(ytr):,} pairs")
    final = fit(kind2, final, Xtr, ytr, X2(es), es["label"].to_numpy())
    del Xtr, ytr, es

    va = lf.filter(inn(va_ids)).collect()
    va = va.with_columns(pl.Series("prob", final.predict_proba(X2(va))[:, 1]))
    t, r = tune_threshold(ent_va, va)
    log(f"FINAL validation  threshold={t}  macro F0.5={r['macro_f05']:.4f}  "
        f"singleton_acc={r['singleton_acc']:.4f}  matched F0.5={r['matched_f05']:.4f}")

    # ---------------------------------------------------------------- phase 5: calibrate + decode
    iso = fit_calibrator(va["prob"].to_numpy(), va["label"].to_numpy())
    va = va.with_columns(pl.Series("p_cal", apply_calibrator(iso, va["prob"].to_numpy())))
    al, eb, fl, r_dec = tune_decoder(va, ent_va, macro_f05_frame)
    log(f"DECODER  alpha={al} empty_boost={eb} floor={fl}  macro F0.5={r_dec['macro_f05']:.4f}  "
        f"singleton_acc={r_dec['singleton_acc']:.4f}  (threshold baseline {r['macro_f05']:.4f})")
    use_dec = r_dec["macro_f05"] > r["macro_f05"]
    log(f"decoder {'ON' if use_dec else 'OFF'}  ({r_dec['macro_f05'] - r['macro_f05']:+.4f} vs threshold)")

    # does hard exclusivity help on top?
    sel = decode(va, "p_cal", al, eb, floor=fl).filter("keep") if use_dec else va.filter(pl.col("prob") >= t)
    kept = enforce_exclusivity(sel)
    r_x = macro_f05_frame(ent_va, va.join(kept.select("idx", "idx2").with_columns(pl.lit(True).alias("p")),
                                          on=["idx", "idx2"], how="left").with_columns(pl.col("p").fill_null(False)), "p")
    use_excl = r_x["macro_f05"] > r["macro_f05"]
    log(f"exclusivity-enforced macro F0.5={r_x['macro_f05']:.4f} -> {'ON' if use_excl else 'OFF'}")

    best = max(r["macro_f05"], r_dec["macro_f05"], r_x["macro_f05"])
    log(f"BEST validation macro F0.5={best:.4f}   GAIN over base: {best - r_b['macro_f05']:+.4f}")

    if kind2 == "lightgbm":
        imp = sorted(zip(FEATURES + COMP_FEATURES, final.booster_.feature_importance("gain")), key=lambda x: -x[1])
        tot = sum(v for _, v in imp) or 1
        print("  top features:", ", ".join(f"{k}={v / tot:.3f}" for k, v in imp[:15]))
        print("  cross-record share:",
              f"{sum(v for k, v in imp if k in COMP_FEATURES) / tot:.3f}")

    cfg = {k: getattr(a, k) for k in ("K", "K_pre", "K_b", "K_pre_b", "max_df", "budget", "no_spell", "n_anchor")}
    joblib.dump({"base": base, "final": final, "kind": kind2, "threshold": t,
                 "calibrator": iso, "alpha": al, "empty_boost": eb, "floor": fl, "use_decoder": bool(use_dec),
                 "threshold_base": t_b, "exclusivity": bool(use_excl),
                 "features": FEATURES, "comp_features": COMP_FEATURES, **cfg}, f"{a.models}/phase5.joblib")
    with open(f"{a.models}/phase5_report.json", "w") as fh:
        json.dump({"recall_ceiling": rc, "base_val": r_b, "final_val": r, "excl_val": r_x,
                   "decoder_val": r_dec, "alpha": al, "empty_boost": eb, "floor": fl, "use_decoder": bool(use_dec),
                   "exclusivity": bool(use_excl), "threshold": t, "exclusivity_holds": shared == 0,
                   "args": vars(a)}, fh, indent=2)
    for f_ in os.listdir(tmp):
        os.remove(os.path.join(tmp, f_))
    os.rmdir(tmp)
    log(f"saved -> {a.models}/phase5.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--s1-sample", type=int, default=250_000)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--K-pre", dest="K_pre", type=int, default=150)
    ap.add_argument("--K-b", dest="K_b", type=int, default=20)
    ap.add_argument("--K-pre-b", dest="K_pre_b", type=int, default=50)
    ap.add_argument("--n-anchor", type=int, default=2, help="top candidates used as transitivity anchors")
    ap.add_argument("--max-df", type=int, default=10_000)
    ap.add_argument("--budget", type=int, default=3_000)
    ap.add_argument("--no-spell", action="store_true")
    ap.add_argument("--batch", type=int, default=750_000, help="rows per normalization batch (lower = less RAM)")
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--neg-sample", type=float, default=0.3,
                    help="fraction of negative pairs used when fitting (1.0 = all; lower = less RAM)")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
