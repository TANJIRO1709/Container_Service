"""Phase 4 inference -> output/matching_results.tsv + output/candidate_pairs.tsv

Two passes over the test blocking (the second is needed because competition features
depend on every S1 entity's scores, which are only known after the first pass):
  pass 1  features -> base model -> keep only (idx, idx2, p_base)   [small in memory]
  pass 2  features again + competition + transitivity -> final model -> write output

Usage (from student_resource/):
  py src\\phase4_predict.py --data dataset --models models --out output --batch 250000
"""
import argparse
import os
import time

import joblib
import numpy as np
import polars as pl

from competition import add_competition, add_support, enforce_exclusivity, target_stats
from io_utils import read_tsv, write_id_lists
from phase4_train import X2
from pipeline3 import Corpus3, X_of


def main(a):
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    art = joblib.load(f"{a.models}/phase4.joblib")
    base, final = art["base"], art["final"]
    thr = art["threshold"] if a.threshold is None else a.threshold
    excl = art["exclusivity"] if a.exclusivity is None else bool(a.exclusivity)
    d = f"{a.data}/test"
    s1 = read_tsv(f"{d}/test_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/test_source2.tsv"), read_tsv(f"{d}/test_source3.tsv")])
    log(f"S1={s1.height:,} targets={tgt.height:,} threshold={thr} exclusivity={excl}")

    corpus = Corpus3(s1, tgt, max_df=art["max_df"], budget=art["budget"], spell=not art["no_spell"], batch=a.batch)
    log("index built")
    it = lambda: corpus.iter_chunks(art["K"], art["K_pre"], a.chunk, a.cost_cap, art["K_b"], art["K_pre_b"])

    # ---- pass 1: base probabilities only
    probs = []
    for i, f in enumerate(it()):
        probs.append(f.select("idx", "idx2").with_columns(
            pl.Series("prob", base.predict_proba(X_of(f))[:, 1]).cast(pl.Float32)))
        if i % 20 == 0:
            log(f"  pass1 chunk {i}")
    probs = pl.concat(probs, rechunk=False)
    log(f"pass 1 done: {probs.height:,} pairs")
    tstats = target_stats(probs)
    log(f"competition stats for {tstats.height:,} targets")

    # ---- pass 2: full features + cross-record, final model
    s1_ids, tg_ids = corpus.s1["entity_id"], corpus.tg["entity_id"]
    cand, match = {}, {}
    for i, f in enumerate(it()):
        f = f.join(probs.rename({"prob": "p_base"}), on=["idx", "idx2"], how="left") \
             .with_columns(pl.col("p_base").fill_null(0.0))
        f = add_support(add_competition(f, tstats), corpus.tg, art["n_anchor"])
        f = f.select("idx", "idx2").with_columns(
            pl.Series("prob", final.predict_proba(X2(f))[:, 1]))
        keep = f.filter(pl.col("prob") >= thr)
        if excl and keep.height:
            keep = enforce_exclusivity(keep)
        f = f.sort(["idx", "prob"], descending=[False, True])
        for sid, tids in f.with_columns(s1_ids.gather(f["idx"]).alias("sid"), tg_ids.gather(f["idx2"]).alias("tid")) \
                          .group_by("sid", maintain_order=True).agg("tid").iter_rows():
            cand[sid] = tids
        keep = keep.sort(["idx", "prob"], descending=[False, True])
        if keep.height:
            for sid, tids in keep.with_columns(s1_ids.gather(keep["idx"]).alias("sid"),
                                               tg_ids.gather(keep["idx2"]).alias("tid")) \
                                 .group_by("sid", maintain_order=True).agg("tid").iter_rows():
                match[sid] = tids
        if i % 20 == 0:
            log(f"  pass2 chunk {i}")

    os.makedirs(a.out, exist_ok=True)
    order = s1["entity_id"].to_list()
    write_id_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", order, cand)
    write_id_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", order, match)
    n = sum(1 for v in match.values() if v)
    log(f"S1 with >=1 match: {n:,}/{len(order):,} ({n / len(order):.3f}); wrote {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="output")
    ap.add_argument("--batch", type=int, default=750_000)
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--exclusivity", type=int, default=None, help="1/0 to override the learned setting")
    main(ap.parse_args())
