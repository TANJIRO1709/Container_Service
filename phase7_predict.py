"""Phase 7 inference -> output/matching_results.tsv + output/candidate_pairs.tsv

  * ONE blocking pass (Phase 5/6 needed two): each chunk's (idx, idx2, prob) is written
    to --parts-dir as soon as it is scored.
  * RESUMABLE: rerun the same command after a disconnect and finished chunks are skipped.
    On Colab, point --parts-dir at Google Drive so the parts survive a runtime reset.
  * Decoding, calibration and exclusivity run GLOBALLY after the pass (Phases 4-6 only
    enforced exclusivity inside each chunk, letting cross-chunk conflicts through).

Usage (from student_resource/):
  python src/phase7_predict.py --data dataset --models models_p7 --out output
  # Colab, resumable across disconnects:
  python src/phase7_predict.py --data dataset --models models_p7 --out output \\
      --parts-dir /content/drive/MyDrive/Amazon_ML/p7_parts
"""
import argparse
import gc
import json
import os
import time

import joblib
import numpy as np
import polars as pl

from competition import enforce_exclusivity
from decode import apply_calibrator, decode
from io_utils import read_tsv
from phase7_train import predict_ens
from pipeline6 import Corpus6, X_of

SCHEMA = {"idx": pl.UInt32, "idx2": pl.UInt32, "prob": pl.Float32}


def main(a):
    t0 = time.time()
    log = lambda m: print(f"{m}  [{time.time() - t0:.0f}s]", flush=True)
    art = joblib.load(f"{a.models}/phase7.joblib")
    models, mode = art["models"], (a.mode or art["mode"])
    d = f"{a.data}/test"
    s1 = read_tsv(f"{d}/test_source1.tsv")
    tgt = pl.concat([read_tsv(f"{d}/test_source2.tsv"), read_tsv(f"{d}/test_source3.tsv")])
    log(f"S1={s1.height:,} targets={tgt.height:,} mode={mode}")

    corpus = Corpus6(s1, tgt, max_df=art["max_df"], budget=art["budget"], spell=not art["no_spell"],
                     batch=a.batch, ngram_mod=art["ngram_mod"])
    del tgt
    log("index built")

    parts = a.parts_dir or os.path.join(a.out, "_parts")
    os.makedirs(parts, exist_ok=True)
    bounds = list(corpus.chunk_bounds(a.chunk, a.cost_cap))
    manifest = {"n_s1": corpus.s1.height, "n_tg": corpus.tg.height, "bounds": len(bounds),
                "chunk": a.chunk, "cost_cap": a.cost_cap, "models": os.path.abspath(a.models)}
    mpath = os.path.join(parts, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath)) != manifest:
        log("!! parts dir was made with different settings -> clearing it")
        for f in os.listdir(parts):
            os.remove(os.path.join(parts, f))
    json.dump(manifest, open(mpath, "w"))
    done = sum(os.path.exists(os.path.join(parts, f"part_{i:05d}.parquet")) for i in range(len(bounds)))
    log(f"{len(bounds)} chunks, {done} already done (resuming)" if done else f"{len(bounds)} chunks")

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
                pl.Series("prob", predict_ens(models, X_of(f))).cast(pl.Float32)).write_parquet(path + ".tmp")
        os.replace(path + ".tmp", path)          # atomic: a half-written part is never trusted
        if i % 20 == 0:
            log(f"  chunk {i}/{len(bounds)}")

    s1_ids = corpus.s1["entity_id"].to_list()
    tg_ids = corpus.tg["entity_id"]
    del corpus
    gc.collect()

    # tie-break on idx2 so the output is byte-identical however many times the run resumed
    probs = pl.scan_parquet(os.path.join(parts, "part_*.parquet")).collect() \
        .sort(["idx", "prob", "idx2"], descending=[False, True, False])
    log(f"scored {probs.height:,} candidate pairs")

    # ---- global selection: decoder / threshold, then exclusivity across ALL chunks
    kept = []
    idxcol = probs["idx"]
    step = 200_000
    for lo in range(0, len(s1_ids), step):
        a0, b0 = int(idxcol.search_sorted(lo, "left")), int(idxcol.search_sorted(lo + step, "left"))
        b = probs.slice(a0, b0 - a0)
        if b.height == 0:
            continue
        if mode.startswith("decoder"):
            b = b.with_columns(pl.Series("p_cal", apply_calibrator(art["calibrator"], b["prob"].to_numpy())))
            kept.append(decode(b, "p_cal", art["alpha"], art["empty_boost"], floor=art["floor"])
                        .filter("keep").select("idx", "idx2", "prob"))
        else:
            kept.append(b.filter(pl.col("prob") >= art["threshold"]).select("idx", "idx2", "prob"))
    kept = pl.concat(kept) if kept else pl.DataFrame(schema=SCHEMA)
    if mode.endswith("+excl") and kept.height:
        before = kept.height
        kept = enforce_exclusivity(kept)
        log(f"exclusivity removed {before - kept.height:,} conflicting matches")
    kept = kept.sort(["idx", "prob", "idx2"], descending=[False, True, False])

    # ---- stream both files in S1 order
    os.makedirs(a.out, exist_ok=True)
    kidx = kept["idx"]
    n_match = 0
    with open(f"{a.out}/candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc, \
         open(f"{a.out}/matching_results.tsv", "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for lo in range(0, len(s1_ids), step):
            hi = min(lo + step, len(s1_ids))

            def lists(df, col):
                a1, b1 = int(col.search_sorted(lo, "left")), int(col.search_sorted(hi, "left"))
                s = df.slice(a1, b1 - a1)
                if s.height == 0:
                    return {}
                return dict(s.with_columns(tg_ids.gather(s["idx2"]).alias("tid"))
                            .group_by("idx", maintain_order=True).agg("tid").iter_rows())

            cd, md = lists(probs, idxcol), lists(kept, kidx)
            for i in range(lo, hi):
                sid = s1_ids[i]
                fc.write(f"{sid}\t{','.join(cd.get(i, ()))}\n")
                m = md.get(i, ())
                n_match += bool(m)
                fm.write(f"{sid}\t{','.join(m)}\n")
    log(f"S1 with >=1 match: {n_match:,}/{len(s1_ids):,} ({n_match / len(s1_ids):.3f}); wrote {a.out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--models", default="models_p7")
    ap.add_argument("--out", default="output")
    ap.add_argument("--parts-dir", default=None, help="where per-chunk scores are saved (use Drive on Colab)")
    ap.add_argument("--mode", default=None, choices=[None, "threshold", "threshold+excl", "decoder", "decoder+excl"],
                    help="override the selection mode chosen in training")
    ap.add_argument("--batch", type=int, default=750_000)
    ap.add_argument("--cost-cap", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=10_000)
    main(ap.parse_args())
