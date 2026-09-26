"""FINAL inference -> output/matching_results.tsv + output/candidate_pairs.tsv

One resumable blocking pass (per-chunk scores saved to --parts-dir; rerun the same
command after a disconnect and finished chunks are skipped), then the decision policy
chosen in training, applied entity-by-entity, with exclusivity enforced GLOBALLY.

Usage (from student_resource/), Colab GPU runtime:
  python src/final_predict.py --data dataset --models models_final --out output \\
      --parts-dir /content/drive/MyDrive/Amazon_ML/final_parts \\
      --emb-cache /content/drive/MyDrive/Amazon_ML/emb_cache
"""
import argparse
import gc
import json
import os
import time

import joblib
import polars as pl

from competition import enforce_exclusivity
from final_train import X, apply_policy
from io_utils import read_tsv
from phase7_train import predict_ens
from pipeline8 import Corpus8

SCHEMA = {"idx": pl.UInt32, "idx2": pl.UInt32, "prob": pl.Float32}


def main(a):
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    art = joblib.load(f"{a.models}/final.joblib")
    models, feats, cfg = art["models"], art["features"], dict(art["policy"])
    log(f"policy: {cfg}")
    d = f"{a.data}/test"
    s1 = read_tsv(f"{d}/test_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/test_source2.tsv"), read_tsv(f"{d}/test_source3.tsv")])
    log(f"S1={s1.height:,} targets={tgt.height:,}")

    corpus = Corpus8(s1, tgt, dense_model=art["dense_model"], K_d=art["K_d"], emb_cache=a.emb_cache, tag="test",
                     dense_batch=a.dense_batch, log=log, max_df=art["max_df"], budget=art["budget"],
                     spell=not art["no_spell"], batch=a.batch, ngram_mod=art["ngram_mod"],
                     global_x=art.get("global_x", False), K_e=art.get("K_e", 5), max_df_e=art.get("max_df_e", 20))
    del tgt
    log("index built")

    parts = a.parts_dir or os.path.join(a.out, "_parts")
    os.makedirs(parts, exist_ok=True)
    bounds = list(corpus.chunk_bounds(a.chunk, a.cost_cap))
    manifest = {"n_s1": corpus.s1.height, "n_tg": corpus.tg.height, "bounds": len(bounds), "chunk": a.chunk,
                "cost_cap": a.cost_cap, "models": os.path.abspath(a.models), "dense": art["dense_model"]}
    mpath = os.path.join(parts, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath)) != manifest:
        log("!! parts dir was made with different settings -> clearing it")
        for f in os.listdir(parts):
            os.remove(os.path.join(parts, f))
    json.dump(manifest, open(mpath, "w"))
    done = sum(os.path.exists(os.path.join(parts, f"part_{i:05d}.parquet")) for i in range(len(bounds)))
    log(f"{len(bounds)} chunks" + (f", {done} already done (resuming)" if done else ""))

    for i, (lo, hi) in enumerate(bounds):
        path = os.path.join(parts, f"part_{i:05d}.parquet")
        if os.path.exists(path):
            continue
        c = corpus.candidates(lo, hi, art["K"], art["K_pre"], art["K_b"], art["K_pre_b"], art["K_c"], art["K_pre_c"])
        if c is None or c.height == 0:
            pl.DataFrame(schema=SCHEMA).write_parquet(path + ".tmp")
        else:
            f = corpus.features(c)
            f.select(pl.col("idx").cast(pl.UInt32), pl.col("idx2").cast(pl.UInt32)).with_columns(
                pl.Series("prob", predict_ens(models, X(f, feats))).cast(pl.Float32)).write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)
        if i % 20 == 0:
            log(f"  chunk {i}/{len(bounds)}")

    s1_ids = corpus.s1["entity_id"].to_list()
    tg_ids = corpus.tg["entity_id"]
    del corpus
    gc.collect()

    probs = pl.scan_parquet(os.path.join(parts, "part_*.parquet")).collect() \
        .sort(["idx", "prob", "idx2"], descending=[False, True, False])
    log(f"scored {probs.height:,} candidate pairs")

    # entity-local decisions (selection + hurdle) in batches, then GLOBAL exclusivity
    step, kept, idxcol = 200_000, [], probs["idx"]
    local = {**cfg, "excl": False}
    for lo in range(0, len(s1_ids), step):
        a0, b0 = int(idxcol.search_sorted(lo, "left")), int(idxcol.search_sorted(lo + step, "left"))
        b = probs.slice(a0, b0 - a0)
        if b.height:
            kept.append(apply_policy(b, None, local, art["hurdle"], art["calibrator"]))
    kept = pl.concat(kept) if kept else pl.DataFrame(schema=SCHEMA)
    if cfg.get("excl") and kept.height:
        n0 = kept.height
        kept = enforce_exclusivity(kept)
        log(f"global exclusivity removed {n0 - kept.height:,} conflicting matches")
    kept = kept.sort(["idx", "prob", "idx2"], descending=[False, True, False])

    os.makedirs(a.out, exist_ok=True)
    kidx, n_match = kept["idx"], 0
    with open(f"{a.out}/candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc, \
         open(f"{a.out}/matching_results.tsv", "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for lo in range(0, len(s1_ids), step):
            hi = min(lo + step, len(s1_ids))

            def lists(df, col):
                x, y = int(col.search_sorted(lo, "left")), int(col.search_sorted(hi, "left"))
                s = df.slice(x, y - x)
                if s.height == 0:
                    return {}
                return dict(s.with_columns(tg_ids.gather(s["idx2"]).alias("tid"))
                            .group_by("idx", maintain_order=True).agg("tid").iter_rows())

            cd, md = lists(probs, idxcol), lists(kept, kidx)
            for i in range(lo, hi):
                m = md.get(i, ())
                n_match += bool(m)
                fc.write(f"{s1_ids[i]}\t{','.join(cd.get(i, ()))}\n")
                fm.write(f"{s1_ids[i]}\t{','.join(m)}\n")
    log(f"S1 with >=1 match: {n_match:,}/{len(s1_ids):,} ({n_match / len(s1_ids):.3f}); wrote {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models_final")
    ap.add_argument("--out", default="output")
    ap.add_argument("--parts-dir", default=None, help="per-chunk scores (use Drive on Colab -> resumable)")
    ap.add_argument("--emb-cache", default=None, help="embedding cache folder (use Drive on Colab)")
    ap.add_argument("--dense-batch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=750_000)
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    main(ap.parse_args())
