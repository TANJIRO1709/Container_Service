"""Phase 1 inference on the test set -> output/matching_results.tsv + output/candidate_pairs.tsv

Usage (from student_resource/):
  python src/phase1_predict.py --data dataset --models models --out output
"""
import argparse
import os
import time

import joblib
import polars as pl

from io_utils import read_tsv, write_id_lists
from pipeline import Corpus, X_of


def main(a):
    t0 = time.time()
    art = joblib.load(f"{a.models}/phase1.joblib")
    model, thr, K, max_df = art["model"], art["threshold"], art["K"], art["max_df"]
    d = f"{a.data}/test"
    s1 = read_tsv(f"{d}/test_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/test_source2.tsv"), read_tsv(f"{d}/test_source3.tsv")])
    print(f"S1={s1.height:,} targets={tgt.height:,}  threshold={thr}  K={K}")

    corpus = Corpus(s1, tgt, max_df=max_df)
    s1_ids, tg_ids = corpus.s1["entity_id"], corpus.tg["entity_id"]
    cand, match = {}, {}
    for i, f in enumerate(corpus.iter_chunks(K, a.chunk)):
        if f.height == 0:
            continue
        f = f.with_columns(pl.Series("prob", model.predict_proba(X_of(f))[:, 1]),
                           s1_ids.gather(f["idx"]).alias("sid"), tg_ids.gather(f["idx2"]).alias("tid"))
        f = f.sort(["idx", "prob"], descending=[False, True])
        for sid, tids in f.group_by("sid", maintain_order=True).agg("tid").iter_rows():
            cand[sid] = tids
        for sid, tids in f.filter(pl.col("prob") >= thr).group_by("sid", maintain_order=True).agg("tid").iter_rows():
            match[sid] = tids
        print(f"  chunk {i}: {f.height:,} pairs [{time.time() - t0:.0f}s]")

    os.makedirs(a.out, exist_ok=True)
    order = s1["entity_id"].to_list()
    write_id_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", order, cand)
    write_id_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", order, match)
    n_match = sum(1 for v in match.values() if v)
    print(f"\nS1 with >=1 match: {n_match:,}/{len(order):,} ({n_match / len(order):.3f})  "
          f"predicted singletons: {1 - n_match / len(order):.3f}")
    print(f"wrote {a.out}/matching_results.tsv and candidate_pairs.tsv [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="output")
    ap.add_argument("--chunk", type=int, default=50_000)
    main(ap.parse_args())
