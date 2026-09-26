"""FINAL training — the best pipeline, end to end.

  blocking   channels A (words) + B (phonetic/acronym/address composites)
             + C (character n-grams) + D (dense multilingual embeddings, optional)
  matcher    LightGBM on 57 pair features (optionally a seed-ensemble)
  decision   isotonic calibration -> per-entity expected-F0.5 decoder
             -> any-match HURDLE model (protects singletons) -> global exclusivity
Every decision component is switched on ONLY if it improves held-out validation
macro F0.5, so adding a component can never make the chosen configuration worse.

Split of S1 entities: 70% train / 10% "es" (early stopping + hurdle training) / 20% validation.

Usage (from student_resource/), Colab GPU runtime:
  python src/final_train.py --data dataset --models models_final --dense-model intfloat/multilingual-e5-small \\
      --emb-cache /content/drive/MyDrive/Amazon_ML/emb_cache
Without a GPU, omit --dense-model (channel D off).
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
from phase7_train import get_model, miss_report, predict_ens
from pipeline8 import Corpus8

HURDLE_FEATURES = ["p1", "p2", "p3", "gap12", "psum", "n50", "n90", "ncand", "ent5"]


def X(f, feats):
    return f.select(feats).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)


def entity_features(pairs, prob_col="prob"):
    """Per-S1 summary of its candidate probabilities, for the any-match hurdle model."""
    top = pairs.sort(["idx", prob_col], descending=[False, True]).group_by("idx", maintain_order=True).agg(
        pl.col(prob_col).head(5).alias("top"), pl.col(prob_col).sum().alias("psum"),
        (pl.col(prob_col) >= 0.5).sum().cast(pl.Float32).alias("n50"),
        (pl.col(prob_col) >= 0.9).sum().cast(pl.Float32).alias("n90"),
        pl.len().cast(pl.Float32).alias("ncand"))
    top = top.with_columns(
        pl.col("top").list.get(0, null_on_oob=True).fill_null(0).alias("p1"),
        pl.col("top").list.get(1, null_on_oob=True).fill_null(0).alias("p2"),
        pl.col("top").list.get(2, null_on_oob=True).fill_null(0).alias("p3"))
    norm = pl.col("top").list.eval(pl.element() / (pl.element().sum() + 1e-9))
    return top.with_columns(
        (pl.col("p1") - pl.col("p2")).alias("gap12"),
        norm.list.eval(-(pl.element() * (pl.element() + 1e-9).log())).list.sum().alias("ent5"),
    ).drop("top")


def apply_policy(pairs, ent_all, cfg, hurdle=None, iso=None):
    """Return the kept (idx, idx2) rows under a decision config.

    pairs: idx, idx2, prob.  ent_all: idx of every entity (for hurdle features).
    """
    if cfg["select"] == "decoder":
        p = pairs.with_columns(pl.Series("p_cal", apply_calibrator(iso, pairs["prob"].to_numpy())))
        kept = decode(p, "p_cal", cfg["alpha"], cfg["empty_boost"], floor=cfg["floor"]).filter("keep")
    else:
        kept = pairs.filter(pl.col("prob") >= cfg["threshold"])
    kept = kept.select("idx", "idx2", "prob")
    if hurdle is not None and cfg.get("hurdle_t", 0) > 0 and kept.height:
        ef = entity_features(pairs)
        q = hurdle.predict_proba(ef.select(HURDLE_FEATURES).to_numpy().astype(np.float32))[:, 1]
        empty = ef.filter(pl.Series(q) < cfg["hurdle_t"]).select("idx")
        kept = kept.join(empty, on="idx", how="anti")
    if cfg.get("excl") and kept.height:
        kept = enforce_exclusivity(kept)
    return kept


def score(ent, pairs, kept):
    lab = pairs.join(kept.select("idx", "idx2").with_columns(pl.lit(True).alias("p")), on=["idx", "idx2"], how="left") \
               .with_columns(pl.col("p").fill_null(False))
    return macro_f05_frame(ent, lab, "p")


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

    corpus = Corpus8(s1, tgt, dense_model=a.dense_model, K_d=a.K_d, emb_cache=a.emb_cache, tag="train",
                     dense_batch=a.dense_batch, log=log, max_df=a.max_df, budget=a.budget,
                     spell=not a.no_spell, batch=a.batch, ngram_mod=a.ngram_mod,
                     global_x=not a.no_global, K_e=a.K_e, max_df_e=a.max_df_e)
    del s1, tgt
    feats = corpus.feature_list()
    log(f"index built  ({len(feats)} features, dense={'ON' if corpus.dense else 'OFF'})")

    id1 = corpus.s1.select(pl.col("entity_id").alias("source1_entity_id"), "idx")
    id2 = corpus.tg.select(pl.col("entity_id").alias("target_id"), pl.col("idx").alias("idx2"))
    gp = explode_ids(gt, "matched_entity_ids").join(id1, on="source1_entity_id").join(id2, on="target_id") \
        .select("idx", "idx2", pl.lit(1, pl.Int8).alias("label"))
    ent = id1.join(gp.group_by("idx").len("G"), on="idx", how="left").select("idx", pl.col("G").fill_null(0))
    mism = gp.with_columns(corpus.s1["cty"].gather(gp["idx"]).alias("c1"), corpus.tg["cty"].gather(gp["idx2"]).alias("c2")) \
             .filter(pl.col("c1") != pl.col("c2")).height
    log(f"COUNTRY-LABEL MISMATCH among ALL true pairs: {mism:,} of {gp.height:,} ({mism / max(gp.height, 1):.2%})"
        f"  <- matches every same-country channel is blind to; channel E is {'OFF' if a.no_global else 'ON'}")

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
        # train entities: all positives + a sample of negatives.  es/val entities: every row.
        parts.append(f.filter((pl.col("split") >= 1) | (pl.col("label") == 1)
                              | (pl.struct("idx", "idx2").hash(seed=13) % 10000 < neg_cut)))
        if i % 10 == 0:
            log(f"  chunk {i}: {f.height:,} pairs")
    rows = pl.concat(parts, rechunk=False)
    del parts
    allpairs = pl.concat(allpairs, rechunk=False)

    rc = rows["label"].sum() / max(gp.height, 1)
    log(f"candidates/S1={n_rows / corpus.s1.height:.1f}  BLOCKING RECALL CEILING={rc:.4f}")
    pos = rows.filter(pl.col("label") == 1)
    lex = (pos["rank"] < a.K) | (pos["rank_b"] < a.K_b) | (pos["rank_c"] < a.K_c)
    if corpus.dense:
        dd = int(pos.filter(~lex)["d_only"].sum())
        log(f"  true pairs found ONLY by dense channel D: {dd:,} (+{dd / gp.height:.4f} recall)")
    if corpus.gx:
        xx = int(pos["x_only"].sum())
        cm = int((pos["cty_eq"] == 0).sum())
        log(f"  true pairs found ONLY by cross-country channel E: {xx:,} (+{xx / gp.height:.4f} recall); "
            f"true pairs with DIFFERENT country labels in candidates: {cm:,}")
    miss_report(corpus, gp, allpairs, a.models)
    del allpairs

    def sub(k):
        return rows.filter(pl.col("split") == k)

    ent_es = ent.with_columns(split.gather(ent["idx"]).alias("s")).filter(pl.col("s") == 1).drop("s")
    ent_va = ent.with_columns(split.gather(ent["idx"]).alias("s")).filter(pl.col("s") == 2).drop("s")
    va = sub(2)
    oracle = macro_f05_frame(ent_va, va.with_columns((pl.col("label") == 1).alias("p")), "p")
    log(f"ORACLE macro F0.5 = {oracle['macro_f05']:.4f}")

    tr = sub(0)
    es_full = sub(1)
    es_fit = es_full.filter((pl.col("label") == 1) | (pl.struct("idx", "idx2").hash(seed=13) % 10000 < neg_cut))
    Xtr, ytr = X(tr, feats), tr["label"].to_numpy()
    Xes, yes = X(es_fit, feats), es_fit["label"].to_numpy()
    del tr, es_fit, rows
    models, kind = [], None
    for s in range(a.n_models):
        kind, m = get_model(a.n_estimators, a.seed + s)
        log(f"training pair model {s + 1}/{a.n_models} ({kind}) on {len(ytr):,} pairs")
        if kind == "lightgbm":
            import lightgbm as lgb
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(Xtr, ytr)
        models.append(m)
    del Xtr, ytr, Xes, yes

    va = va.with_columns(pl.Series("prob", predict_ens(models, X(va, feats))))
    es_full = es_full.with_columns(pl.Series("prob", predict_ens(models, X(es_full, feats))))
    vp = va.select("idx", "idx2", "prob", "label")

    # ---------------------------------------------------------------- decision policy
    t, _ = tune_threshold(ent_va, va)
    iso = fit_calibrator(va["prob"].to_numpy(), va["label"].to_numpy())
    va = va.with_columns(pl.Series("p_cal", apply_calibrator(iso, va["prob"].to_numpy())))
    al, eb, fl, _ = tune_decoder(va, ent_va, macro_f05_frame)

    # hurdle model: trained on es entities (not used to fit the pair model), label = has any match
    ef = entity_features(es_full.select("idx", "idx2", "prob")).join(ent_es, on="idx", how="right") \
        .fill_null(0)
    hurdle = None
    if ef.height > 500 and (ef["G"] == 0).sum() >= 50:
        import lightgbm as lgb
        hurdle = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                    verbose=-1, random_state=a.seed)
        hurdle.fit(ef.select(HURDLE_FEATURES).to_numpy().astype(np.float32), (ef["G"] > 0).to_numpy().astype(int))
        log(f"hurdle model trained on {ef.height:,} es entities ({int((ef['G'] == 0).sum()):,} singletons)")

    base = {"threshold": t, "alpha": al, "empty_boost": eb, "floor": fl}
    results = {}
    for sel in ("threshold", "decoder"):
        for excl in (False, True):
            for ht in ([0.0] + ([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if hurdle is not None else [])):
                cfg = {**base, "select": sel, "excl": excl, "hurdle_t": ht}
                r = score(ent_va, vp, apply_policy(vp, ent_va, cfg, hurdle, iso))
                results[(sel, excl, ht)] = (cfg, r)
    best_key = max(results, key=lambda k: results[k][1]["macro_f05"])
    cfg, best = results[best_key]
    for sel in ("threshold", "decoder"):
        for excl in (False, True):
            r0 = results[(sel, excl, 0.0)][1]
            hk = max((k for k in results if k[0] == sel and k[1] == excl), key=lambda k: results[k][1]["macro_f05"])
            log(f"  {sel + ('+excl' if excl else ''):16s} F0.5={r0['macro_f05']:.4f}"
                f"   with best hurdle(t={hk[2]}): {results[hk][1]['macro_f05']:.4f}")
    log(f"BEST: select={cfg['select']} excl={cfg['excl']} hurdle_t={cfg['hurdle_t']}  "
        f"VALIDATION macro F0.5={best['macro_f05']:.4f}  singleton_acc={best['singleton_acc']:.4f}  "
        f"(oracle {oracle['macro_f05']:.4f}, model gap {best['macro_f05'] - oracle['macro_f05']:+.4f})")

    if kind == "lightgbm":
        imp = np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)
        o = np.argsort(-imp)
        print("  top features:", ", ".join(f"{feats[i]}={imp[i] / imp.sum():.3f}" for i in o[:12]))

    keep_cfg = {k: getattr(a, k) for k in ("K", "K_pre", "K_b", "K_pre_b", "K_c", "K_pre_c", "K_d", "max_df",
                                            "budget", "no_spell", "ngram_mod", "dense_model", "K_e", "max_df_e")}
    keep_cfg["global_x"] = not a.no_global
    joblib.dump({"models": models, "hurdle": hurdle, "calibrator": iso, "policy": cfg, "features": feats,
                 **keep_cfg}, f"{a.models}/final.joblib")
    with open(f"{a.models}/final_report.json", "w") as fh:
        json.dump({"recall_ceiling": rc, "oracle": oracle, "best": best, "policy": cfg,
                   "all": {f"{k[0]}|excl={k[1]}|hurdle={k[2]}": v[1]["macro_f05"] for k, v in results.items()},
                   "args": vars(a)}, fh, indent=2)
    log(f"saved -> {a.models}/final.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models_final")
    ap.add_argument("--dense-model", default=None,
                    help="e.g. intfloat/multilingual-e5-small (GPU). Omit to disable channel D. 'hash' = CPU test mode")
    ap.add_argument("--K-d", dest="K_d", type=int, default=20, help="dense neighbours per S1")
    ap.add_argument("--emb-cache", default=None, help="folder to cache embeddings (use Drive on Colab)")
    ap.add_argument("--dense-batch", type=int, default=512)
    ap.add_argument("--no-global", action="store_true", help="disable the cross-country fallback channel E")
    ap.add_argument("--K-e", dest="K_e", type=int, default=5, help="cross-country candidates per S1")
    ap.add_argument("--max-df-e", type=int, default=20, help="channel E uses only keys this rare")
    ap.add_argument("--s1-sample", type=int, default=400_000)
    ap.add_argument("--neg-sample", type=float, default=0.3)
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--K-pre", dest="K_pre", type=int, default=150)
    ap.add_argument("--K-b", dest="K_b", type=int, default=20)
    ap.add_argument("--K-pre-b", dest="K_pre_b", type=int, default=50)
    ap.add_argument("--K-c", dest="K_c", type=int, default=20)
    ap.add_argument("--K-pre-c", dest="K_pre_c", type=int, default=50)
    ap.add_argument("--ngram-mod", type=int, default=2)
    ap.add_argument("--max-df", type=int, default=10_000)
    ap.add_argument("--budget", type=int, default=3_000)
    ap.add_argument("--no-spell", action="store_true")
    ap.add_argument("--batch", type=int, default=750_000)
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
