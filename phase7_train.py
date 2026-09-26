"""Phase 7 training — the consolidated "best" pipeline.

What the real-data Phase-6 report showed, and what this changes:
  * base model 0.9404 vs cross-record "final" model 0.9354 (singleton acc 0.927 -> 0.883):
    the second-stage model HURT. Phase 7 uses ONE strong model (optionally a small
    seed-ensemble) and puts calibration + expected-F0.5 decoding + exclusivity on top.
  * recall ceiling 0.9286 was still the biggest loss: denser character n-grams
    (ngram_mod 2) and more channel-C candidates (K_c 20) by default.
  * negatives are subsampled AT COLLECTION TIME for training entities, so more S1
    entities fit in RAM; validation entities keep every row so the score is honest.
  * prints the MISS DIAGNOSTICS so the remaining recall loss can be inspected.

Usage (from student_resource/):
  python src/phase7_train.py --data dataset --models models_p7
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import polars as pl

from competition import enforce_exclusivity
from decode import apply_calibrator, decode, fit_calibrator, tune as tune_decoder
from evaluate import macro_f05_frame, tune_threshold
from io_utils import explode_ids, read_tsv
from pipeline6 import FEATURES, Corpus6, X_of


def get_model(n_estimators, seed):
    try:
        import lightgbm as lgb
        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=255, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbose=-1)
    except (ImportError, OSError) as e:
        print(f"!! LightGBM unavailable ({e}); using sklearn HistGradientBoosting")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hgb", HistGradientBoostingClassifier(
            max_iter=n_estimators, learning_rate=0.08, max_leaf_nodes=255, min_samples_leaf=50,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=seed)


def predict_ens(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def miss_report(corpus, gp, pairs, out_dir, n_sample=3000):
    miss = gp.join(pairs, on=["idx", "idx2"], how="anti")
    print(f"\n--- MISS DIAGNOSTICS: {miss.height:,} true pairs not retrieved ---")
    if miss.height == 0:
        return
    m = miss.sample(min(n_sample, miss.height), seed=0)
    A, B = corpus.s1, corpus.tg
    g = lambda df, c, i: df[c].gather(m[i]).fill_null("").to_list()
    cols = dict(s1_id=g(A, "entity_id", "idx"), tgt_id=g(B, "entity_id", "idx2"),
                c1=g(A, "cty", "idx"), c2=g(B, "cty", "idx2"),
                n1=g(A, "name_key", "idx"), n2=g(B, "name_key", "idx2"),
                a1=g(A, "addr_key", "idx"), a2=g(B, "addr_key", "idx2"))
    tag = []
    for c1, c2, n1, n2, a1, a2 in zip(cols["c1"], cols["c2"], cols["n1"], cols["n2"], cols["a1"], cols["a2"]):
        sn, sa = set(n1.split()) & set(n2.split()), set(a1.split()) & set(a2.split())
        tag.append("country_mismatch" if c1 != c2 else "empty_name_or_addr" if not (n1 and n2 and a1 and a2)
                   else "shares_nothing" if not sn and not sa else "address_only_overlap" if not sn
                   else "name_only_overlap" if not sa else "name_and_addr_overlap")
    df = pl.DataFrame({**cols, "category": tag})
    for k, v in df.group_by("category").len().sort("len", descending=True).iter_rows():
        print(f"  {k:24s} {v / len(tag):6.1%}")
    for r in df.head(10).iter_rows(named=True):
        print(f"   [{r['category']}] {r['n1']!r} | {r['n2']!r}\n{'':6}{r['a1']!r} | {r['a2']!r}")
    df.write_csv(f"{out_dir}/phase7_missed_sample.tsv", separator="\t")
    print(f"  sample -> {out_dir}/phase7_missed_sample.tsv\n")


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

    corpus = Corpus6(s1, tgt, max_df=a.max_df, budget=a.budget, spell=not a.no_spell,
                     batch=a.batch, ngram_mod=a.ngram_mod)
    del s1, tgt
    log("index built")

    id1 = corpus.s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
    id2 = corpus.tg.select(pl.col("entity_id").alias("target_id"), pl.col("idx").alias("idx2"))
    gp = explode_ids(gt, "matched_entity_ids").join(id1, on="source1_entity_id").join(id2, on="target_id") \
        .select("idx", "idx2", pl.lit(1, pl.Int8).alias("label"))
    ent = id1.join(gp.group_by("idx").len("G"), on="idx", how="left").select("idx", pl.col("G").fill_null(0))

    # split by S1 entity: 0 train / 1 early-stopping / 2 validation
    u = np.random.default_rng(a.seed).random(corpus.s1.height)
    split = pl.Series("split", np.where(u < 0.7, 0, np.where(u < 0.8, 1, 2)).astype(np.int8))
    neg_cut = int(a.neg_sample * 10000)

    parts, allpairs, n_rows = [], [], 0
    for i, (lo, hi) in enumerate(corpus.chunk_bounds(a.chunk, a.cost_cap)):
        c = corpus.candidates(lo, hi, a.K, a.K_pre, a.K_b, a.K_pre_b, a.K_c, a.K_pre_c)
        if c is None or c.height == 0:
            continue
        f = corpus.features(c).join(gp, on=["idx", "idx2"], how="left").with_columns(pl.col("label").fill_null(0))
        f = f.with_columns(split.gather(f["idx"]).alias("split"))
        n_rows += f.height
        allpairs.append(f.select("idx", "idx2"))
        keep = (pl.col("split") == 2) | (pl.col("label") == 1) \
            | (pl.struct("idx", "idx2").hash(seed=13) % 10000 < neg_cut)
        parts.append(f.filter(keep))
        if i % 10 == 0:
            log(f"  chunk {i}: {f.height:,} pairs")
    rows = pl.concat(parts, rechunk=False)
    del parts
    allpairs = pl.concat(allpairs, rechunk=False)

    rc = rows["label"].sum() / max(gp.height, 1)
    log(f"candidates/S1={n_rows / corpus.s1.height:.1f}  BLOCKING RECALL CEILING={rc:.4f}")
    pos = rows.filter(pl.col("label") == 1)
    ob = int(((pos["rank"] >= a.K) & (pos["rank_c"] >= a.K_c)).sum())
    oc = int(((pos["rank"] >= a.K) & (pos["rank_b"] >= a.K_b)).sum())
    log(f"  reachable ONLY via channel B: +{ob / gp.height:.4f}   ONLY via channel C: +{oc / gp.height:.4f}")
    miss_report(corpus, gp, allpairs, a.models)
    del allpairs

    ent_va = ent.with_columns(split.gather(ent["idx"]).alias("split")).filter(pl.col("split") == 2).drop("split")
    va = rows.filter(pl.col("split") == 2)
    oracle = macro_f05_frame(ent_va, va.with_columns((pl.col("label") == 1).alias("p")), "p")
    log(f"ORACLE macro F0.5 (perfect model on these candidates) = {oracle['macro_f05']:.4f}")

    tr, es = rows.filter(pl.col("split") == 0), rows.filter(pl.col("split") == 1)
    del rows
    Xtr, ytr, Xes, yes = X_of(tr), tr["label"].to_numpy(), X_of(es), es["label"].to_numpy()
    del tr, es
    models, kind = [], None
    for s in range(a.n_models):
        kind, m = get_model(a.n_estimators, a.seed + s)
        log(f"training model {s + 1}/{a.n_models} ({kind}) on {len(ytr):,} pairs")
        if kind == "lightgbm":
            import lightgbm as lgb
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(Xtr, ytr)
        models.append(m)
    del Xtr, ytr, Xes, yes

    va = va.with_columns(pl.Series("prob", predict_ens(models, X_of(va))))
    t, r_t = tune_threshold(ent_va, va)
    iso = fit_calibrator(va["prob"].to_numpy(), va["label"].to_numpy())
    va = va.with_columns(pl.Series("p_cal", apply_calibrator(iso, va["prob"].to_numpy())))
    al, eb, fl, r_d = tune_decoder(va, ent_va, macro_f05_frame)

    def with_excl(sel):
        kept = enforce_exclusivity(sel.select("idx", "idx2", "prob"))
        return macro_f05_frame(ent_va, va.join(kept.select("idx", "idx2").with_columns(pl.lit(True).alias("p")),
                                               on=["idx", "idx2"], how="left")
                               .with_columns(pl.col("p").fill_null(False)), "p")

    r_tx = with_excl(va.filter(pl.col("prob") >= t))
    r_dx = with_excl(decode(va, "p_cal", al, eb, floor=fl).filter("keep"))
    options = {"threshold": r_t, "threshold+excl": r_tx, "decoder": r_d, "decoder+excl": r_dx}
    for k, v in options.items():
        log(f"  {k:16s} macro F0.5={v['macro_f05']:.4f}  singleton_acc={v['singleton_acc']:.4f}")
    mode = max(options, key=lambda k: options[k]["macro_f05"])
    best = options[mode]
    log(f"BEST: {mode}  validation macro F0.5={best['macro_f05']:.4f}   (oracle {oracle['macro_f05']:.4f}, "
        f"model gap {best['macro_f05'] - oracle['macro_f05']:+.4f})")

    if kind == "lightgbm":
        imp = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
        order = np.argsort(-imp)
        print("  top features:", ", ".join(f"{FEATURES[i]}={imp[i] / imp.sum():.3f}" for i in order[:12]))

    cfg = {k: getattr(a, k) for k in ("K", "K_pre", "K_b", "K_pre_b", "K_c", "K_pre_c", "max_df", "budget",
                                       "no_spell", "ngram_mod")}
    joblib.dump({"models": models, "kind": kind, "mode": mode, "threshold": t, "calibrator": iso,
                 "alpha": al, "empty_boost": eb, "floor": fl, "features": FEATURES, **cfg},
                f"{a.models}/phase7.joblib")
    with open(f"{a.models}/phase7_report.json", "w") as fh:
        json.dump({"recall_ceiling": rc, "oracle": oracle, "mode": mode, "best": best,
                   "options": options, "threshold": t, "alpha": al, "empty_boost": eb, "floor": fl,
                   "args": vars(a)}, fh, indent=2)
    log(f"saved -> {a.models}/phase7.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models_p7")
    ap.add_argument("--s1-sample", type=int, default=400_000, help="S1 entities used (0 = all)")
    ap.add_argument("--neg-sample", type=float, default=0.3, help="fraction of negatives kept for TRAINING entities")
    ap.add_argument("--n-models", type=int, default=1, help="seed-ensemble size (2-3 helps a little)")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--K-pre", dest="K_pre", type=int, default=150)
    ap.add_argument("--K-b", dest="K_b", type=int, default=20)
    ap.add_argument("--K-pre-b", dest="K_pre_b", type=int, default=50)
    ap.add_argument("--K-c", dest="K_c", type=int, default=20)
    ap.add_argument("--K-pre-c", dest="K_pre_c", type=int, default=50)
    ap.add_argument("--ngram-mod", type=int, default=2, help="keep 1/N of char n-grams (lower = more recall, slower)")
    ap.add_argument("--max-df", type=int, default=10_000)
    ap.add_argument("--budget", type=int, default=3_000)
    ap.add_argument("--no-spell", action="store_true")
    ap.add_argument("--batch", type=int, default=750_000)
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
