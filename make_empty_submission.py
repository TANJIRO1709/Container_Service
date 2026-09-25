"""Submission #1 - the singleton-rate probe.
Every S1 entity gets an empty list, so the leaderboard score equals the
fraction of true singletons in the public test subset.

Usage (from student_resource/):
  python src/make_empty_submission.py --data dataset --out output
  python3 utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
import argparse
import os

from io_utils import scan_tsv, write_id_lists

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--out", default="output")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
ids = scan_tsv(f"{a.data}/test/test_source1.tsv").select("entity_id").collect()["entity_id"]
assert ids.n_unique() == len(ids), "duplicate S1 ids in test file?"
ids = ids.to_list()
write_id_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", ids, {})
write_id_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", ids, {})
print(f"wrote {len(ids):,} rows to {a.out}/")
