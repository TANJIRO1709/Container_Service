"""Phase-1 EDA: answers the structural questions that decide the design.

Usage (from student_resource/):
  python src/eda.py --data dataset | tee eda_report.txt
"""
import argparse
import time

import polars as pl

from io_utils import explode_ids, read_tsv, scan_tsv


def hr(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def norm(col):
    return pl.col(col).fill_null("").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ") \
        .str.replace_all(r"\s+", " ").str.strip_chars()


def main(d):
    t0 = time.time()
    pl.Config.set_tbl_rows(40)
    pl.Config.set_fmt_str_lengths(80)
    pl.Config.set_tbl_width_chars(200)

    # 1. Sizes and countries (lazy, so test files are never fully loaded)
    hr("1. ROW COUNTS x COUNTRY (all files)")
    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = f"{d}/{split}/{split}_source{s}.tsv"
            c = scan_tsv(p).group_by("country").len().sort("len", descending=True).collect()
            print(f"{split}_source{s}: total={c['len'].sum():,}  " +
                  "  ".join(f"{r[0]}={r[1]:,}" for r in c.iter_rows()))

    # Load train
    s1 = read_tsv(f"{d}/train/train_source1.tsv")
    s2 = read_tsv(f"{d}/train/train_source2.tsv")
    s3 = read_tsv(f"{d}/train/train_source3.tsv")
    gt = read_tsv(f"{d}/train/train_ground_truth.tsv")
    s23 = pl.concat([s2, s3])
    print(f"\n[loaded train in {time.time() - t0:.1f}s]")

    hr("2. NULL / EMPTY RATES")
    for name, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        rates = {c: round(df.select((pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")).mean()).item(), 4)
                 for c in ("business_name", "business_address", "country")}
        print(name, rates)

    hr("3. GROUND TRUTH: coverage, singletons, match-count distribution")
    print(f"GT rows={gt.height:,}  unique S1 in GT={gt['source1_entity_id'].n_unique():,}  S1 rows={s1.height:,}")
    missing = s1.filter(~pl.col("entity_id").is_in(gt["source1_entity_id"].implode())).height
    print(f"S1 ids missing from GT: {missing:,}")
    pairs = explode_ids(gt, "matched_entity_ids")
    cnt = gt.select("source1_entity_id").join(
        pairs.group_by("source1_entity_id").len(), on="source1_entity_id", how="left"
    ).with_columns(pl.col("len").fill_null(0))
    print(f"Singleton rate: {(cnt['len'] == 0).mean():.4f}")
    dist = cnt.with_columns(pl.when(pl.col("len") >= 6).then(6).otherwise(pl.col("len")).alias("k")) \
        .group_by("k").len().sort("k").with_columns((pl.col("len") / cnt.height).round(4).alias("frac"))
    print("match count per S1 (6 = 6+):\n", dist)
    print(f"max matches for one S1: {cnt['len'].max()}  mean(non-singleton): "
          f"{cnt.filter(pl.col('len') > 0)['len'].mean():.3f}")

    # By country
    cc = cnt.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "country"), on="source1_entity_id")
    print("\nsingleton rate & mean matches by S1 country:\n",
          cc.group_by("country").agg(pl.len().alias("n"), (pl.col("len") == 0).mean().round(4).alias("singleton_rate"),
                                     pl.col("len").mean().round(3).alias("mean_matches")))

    hr("4. PAIR COMPOSITION & EXCLUSIVITY (pillar-1 check)")
    pairs = pairs.with_columns(pl.col("target_id").str.slice(0, 2).alias("src"))
    print("pairs by source:", dict(pairs.group_by("src").len().iter_rows()))
    per = pairs.group_by("source1_entity_id", "src").len()
    print("records per S1 per source (dupes inside S2/S3?):\n",
          per.group_by("src").agg((pl.col("len") > 1).mean().round(4).alias("frac_S1_with_>1"),
                                  pl.col("len").max().alias("max")))
    multi = pairs.group_by("target_id").agg(pl.col("source1_entity_id").n_unique().alias("n_s1")) \
        .filter(pl.col("n_s1") > 1)
    print(f"S2/S3 ids matched to >1 S1: {multi.height:,} of {pairs['target_id'].n_unique():,} "
          f"-> exclusivity {'HOLDS' if multi.height == 0 else 'VIOLATED (inspect!)'}")
    ids23 = set(s23["entity_id"].to_list())
    bad = pairs.filter(~pl.col("target_id").is_in(list(ids23))).height
    print(f"GT ids not found in S2/S3 files: {bad:,}")
    matched23 = pairs["target_id"].n_unique()
    print(f"S2 records matched: {pairs.filter(pl.col('src') == 'S2')['target_id'].n_unique() / s2.height:.4f}  "
          f"S3 records matched: {pairs.filter(pl.col('src') == 'S3')['target_id'].n_unique() / s3.height:.4f}  "
          f"(rest are distractors)")

    hr("5. COUNTRY AGREEMENT & RAW NOISE LEVEL in true pairs")
    j = pairs.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id") \
        .join(s23.rename({"entity_id": "target_id"}), on="target_id", suffix="_t")
    print(f"country mismatch rate in true pairs: {(j['country'] != j['country_t']).mean():.5f}")
    j = j.with_columns(norm("business_name").alias("n1"), norm("business_name_t").alias("n2"),
                       norm("business_address").alias("a1"), norm("business_address_t").alias("a2"))
    print(f"exact normalized NAME equal: {(j['n1'] == j['n2']).mean():.4f}   "
          f"exact normalized ADDRESS equal: {(j['a1'] == j['a2']).mean():.4f}")
    # Name collisions among S1: how ambiguous are names alone?
    s1n = s1.with_columns(norm("business_name").alias("n"))
    dup = s1n.group_by("country", "n").len().filter(pl.col("len") > 1)
    print(f"S1 records sharing an exact normalized name with another S1: "
          f"{dup['len'].sum() / s1.height:.4f}  (name alone is {'NOT ' if dup.height else ''}unique)")

    hr("6. LENGTHS (tokens)")
    for name, df in (("S1", s1), ("S2", s2), ("S3", s3)):
        print(name, df.select(
            norm("business_name").str.split(" ").list.len().mean().round(2).alias("name_tok"),
            norm("business_address").str.split(" ").list.len().mean().round(2).alias("addr_tok")).row(0))

    hr("7. SAMPLE TRUE PAIRS (eyeball the noise)")
    for c in j["country"].unique().to_list():
        print(f"\n--- {c} ---")
        smp = j.filter(pl.col("country") == c).sample(min(12, j.filter(pl.col("country") == c).height), seed=0)
        for r in smp.select("business_name", "business_name_t", "business_address", "business_address_t").iter_rows():
            print(f"  N: {r[0]!r:45} | {r[1]!r}\n  A: {r[2]!r:45} | {r[3]!r}\n")

    hr("8. TEST: sample France rows")
    fr = scan_tsv(f"{d}/test/test_source1.tsv").filter(pl.col("country").str.to_lowercase().str.contains("fr")) \
        .head(10).collect()
    for r in fr.select("business_name", "business_address", "country").iter_rows():
        print(" ", r)
    print(f"\n[done in {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    main(ap.parse_args().data)
