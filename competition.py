"""Phase 4: cross-record features — competition, exclusivity, and S2<->S3 transitivity.

Phase 1-3 score each (S1, target) pair in isolation. Two facts that a pair-only model
cannot see:

1. COMPETITION. Source 1 is deduplicated, so a given S2/S3 record belongs to at most one
   real S1 entity. A candidate that is *also* the best match for three other S1 entities
   is far more likely to be a false merge (chain branch, common name) than one that only
   this S1 wants. Features: how many S1 entities compete for this target, this pair's
   rank among them, its margin behind the leader, and whether the pair is mutually best.

2. TRANSITIVITY. Each real business forms a cluster across the sources. If S2-x looks
   only moderately similar to the S1 record, but is nearly identical to S3-y — which
   matches the S1 record strongly — then S2-x belongs to the cluster too. Features:
   similarity of the candidate to the S1's top-scoring candidates ("anchors"), weighted
   by how confident those anchors are.

Both are computed from a cheap first-pass probability, so no extra blocking pass is
needed: pass 1 scores pairs with the base model, this module derives the cross-record
signals, pass 2 rescores with them included.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

# features added on top of the base (pipeline3) feature list
COMP_FEATURES = [
    "p_base", "p_rank", "p_max_s1", "p_gap_s1", "p_sum_s1",
    "comp_n", "comp_rank", "comp_best", "comp_margin", "comp_ratio", "mutual_best",
    "sup_name", "sup_addr", "sup_prob", "sup_combo", "anchor_same_src",
]

P_FLOOR = 0.05   # ignore near-zero pairs when counting competitors


def target_stats(probs: pl.DataFrame) -> pl.DataFrame:
    """Global per-target (idx2) competition stats from pass-1 probabilities.

    probs: (idx, idx2, prob) for every candidate pair in the corpus.
    Returns one row per idx2: how many S1 entities want it, and the top two scores.
    """
    live = probs.filter(pl.col("prob") >= P_FLOOR)
    return live.group_by("idx2").agg(
        pl.len().cast(pl.Float32).alias("comp_n"),
        pl.col("prob").max().alias("comp_best"),
        pl.col("prob").top_k(2).last().alias("comp_second"),
        pl.col("idx").sort_by("prob", descending=True).first().alias("comp_winner"),
    )


def add_competition(f: pl.DataFrame, tstats: pl.DataFrame) -> pl.DataFrame:
    """Attach competition features to a chunk of candidate rows that already carry p_base."""
    f = f.join(tstats, on="idx2", how="left").with_columns(
        pl.col("comp_n").fill_null(0.0), pl.col("comp_best").fill_null(0.0),
        pl.col("comp_second").fill_null(0.0),
    )
    return f.with_columns(
        # rank within this S1 entity
        pl.col("p_base").rank("ordinal", descending=True).over("idx").cast(pl.Float32).alias("p_rank"),
        pl.col("p_base").max().over("idx").alias("p_max_s1"),
        pl.col("p_base").sum().over("idx").alias("p_sum_s1"),
        # rank among the S1 entities competing for this target
        (pl.col("p_base") < pl.col("comp_best")).cast(pl.Float32).alias("comp_rank"),
        (pl.col("comp_winner") == pl.col("idx")).fill_null(False).cast(pl.Float32).alias("mutual_best"),
    ).with_columns(
        (pl.col("p_max_s1") - pl.col("p_base")).alias("p_gap_s1"),
        # margin: leader's lead over the runner-up if we ARE the leader, else our deficit
        pl.when(pl.col("mutual_best") > 0)
        .then(pl.col("comp_best") - pl.col("comp_second"))
        .otherwise(pl.col("p_base") - pl.col("comp_best")).alias("comp_margin"),
        (pl.col("p_base") / pl.when(pl.col("comp_best") > 0).then(pl.col("comp_best")).otherwise(1.0))
        .alias("comp_ratio"),
    ).drop("comp_second", "comp_winner")


def add_support(f: pl.DataFrame, tg: pl.DataFrame, n_anchor: int = 2, sim_batch: int = 2_000_000) -> pl.DataFrame:
    """S2<->S3 transitivity: similarity of each candidate to this S1's top anchors.

    All candidates of one S1 entity live in the same chunk, so this is chunk-local.
    """
    empty = {c: 0.0 for c in ("sup_name", "sup_addr", "sup_prob", "sup_combo", "anchor_same_src")}
    if f.height == 0:
        return f.with_columns([pl.lit(v, pl.Float32).alias(k) for k, v in empty.items()])
    anchors = (f.select("idx", "idx2", "p_base")
               .filter(pl.col("p_base") >= P_FLOOR)
               .sort(["idx", "p_base"], descending=[False, True])
               .filter(pl.int_range(pl.len()).over("idx") < n_anchor)
               .rename({"idx2": "a_idx2", "p_base": "a_prob"}))
    if anchors.height == 0:
        return f.with_columns([pl.lit(v, pl.Float32).alias(k) for k, v in empty.items()])
    pairs = f.select("idx", "idx2").join(anchors, on="idx").filter(pl.col("idx2") != pl.col("a_idx2"))
    if pairs.height == 0:
        return f.with_columns([pl.lit(v, pl.Float32).alias(k) for k, v in empty.items()])

    kw = dict(workers=-1, dtype=np.float32)
    src = tg["entity_id"].str.slice(0, 2)
    # batched: materialising every pair's strings as Python lists at once is the peak
    sn, sa, ss = [], [], []
    for o in range(0, pairs.height, sim_batch):
        p = pairs.slice(o, sim_batch)
        i, j = p["idx2"], p["a_idx2"]
        gl = lambda col, ix: tg[col].gather(ix).fill_null("").to_list()
        sn.append(process.cpdist(gl("name_key", i), gl("name_key", j), scorer=fuzz.token_set_ratio, **kw))
        sa.append(process.cpdist(gl("addr_key", i), gl("addr_key", j), scorer=fuzz.token_set_ratio, **kw))
        ss.append((src.gather(i) == src.gather(j)).cast(pl.Float32).to_numpy())
    pairs = pairs.with_columns(
        pl.Series("sn", np.concatenate(sn)), pl.Series("sa", np.concatenate(sa)),
        pl.Series("same_src", np.concatenate(ss)),
    ).with_columns(((pl.col("sn") + pl.col("sa")) / 200.0 * pl.col("a_prob")).alias("combo"))

    agg = pairs.group_by("idx", "idx2").agg(
        pl.col("sn").max().alias("sup_name"), pl.col("sa").max().alias("sup_addr"),
        pl.col("a_prob").max().alias("sup_prob"), pl.col("combo").max().alias("sup_combo"),
        pl.col("same_src").filter(pl.col("combo") == pl.col("combo").max()).first().alias("anchor_same_src"),
    )
    return f.join(agg, on=["idx", "idx2"], how="left").with_columns(
        [pl.col(k).fill_null(0.0) for k in empty])


def enforce_exclusivity(pred: pl.DataFrame, prob_col: str = "prob") -> pl.DataFrame:
    """Keep each target for the single S1 entity that scores it highest.

    Source 1 is deduplicated, so a target belongs to at most one S1 entity. Only applied
    when a config flag asks for it — verify on your own ground truth first (the training
    check is printed by phase4_train.py).
    """
    best = pred.group_by("idx2").agg(pl.col("idx").sort_by(prob_col, descending=True).first().alias("w"))
    return pred.join(best, on="idx2", how="left").filter(
        pl.col("w").is_null() | (pl.col("w") == pl.col("idx"))).drop("w")
