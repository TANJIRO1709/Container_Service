"""Phase 2 inference -> output/matching_results.tsv + output/candidate_pairs.tsv

Usage (from student_resource/):
  python3 src/phase2_predict.py --data dataset --models models --out output
"""
import argparse
import os
import time

import joblib
import polars as pl

from io_utils import read_tsv, write_id_lists
from pipeline2 import Corpus2, X_of


def main(a):
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    art = joblib.load(f"{a.models}/phase2.joblib")
    model, thr = art["model"], art["threshold"] if a.threshold is None else a.threshold
    d = f"{a.data}/test"
    s1 = read_tsv(f"{d}/test_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/test_source2.tsv"), read_tsv(f"{d}/test_source3.tsv")])
    log(f"S1={s1.height:,} targets={tgt.height:,} threshold={thr} K={art['K']}")
    corpus = Corpus2(s1, tgt, max_df=art["max_df"], budget=art["budget"], spell=not art["no_spell"])
    log("index built")
    s1_ids, tg_ids = corpus.s1["entity_id"], corpus.tg["entity_id"]
    cand, match = {}, {}
    for i, f in enumerate(corpus.iter_chunks(art["K"], art["K_pre"], a.chunk, a.cost_cap)):
        f = f.select("idx", "idx2").with_columns(
            pl.Series("prob", model.predict_proba(X_of(f))[:, 1]),
            s1_ids.gather(f["idx"]).alias("sid"), tg_ids.gather(f["idx2"]).alias("tid"),
        ).sort(["idx", "prob"], descending=[False, True])
        for sid, tids in f.group_by("sid", maintain_order=True).agg("tid").iter_rows():
            cand[sid] = tids
        for sid, tids in f.filter(pl.col("prob") >= thr).group_by("sid", maintain_order=True).agg("tid").iter_rows():
            match[sid] = tids
        if i % 10 == 0:
            log(f"  chunk {i}: {f.height:,} pairs")
    os.makedirs(a.out, exist_ok=True)
    order = s1["entity_id"].to_list()
    write_id_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", order, cand)
    write_id_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", order, match)
    n_match = sum(1 for v in match.values() if v)
    log(f"S1 with >=1 match: {n_match:,}/{len(order):,} ({n_match / len(order):.3f}); wrote {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cost-cap", type=int, default=20_000_000, help="max retrieval rows per chunk (lower = less RAM)")
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    main(ap.parse_args())
