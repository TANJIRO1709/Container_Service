"""Phase 2 core.

What changed vs Phase 1
-----------------------
1. Typo-aware keys: rare tokens are mapped to their frequent edit-distance-1 neighbour
   (spell.py) before keys are built, so "Shamra Eage Logitics" now shares keys with
   "Sharma Eagle Logistics".
2. Budgeted rare-key retrieval (fixes the max_df=500 recall cap at 10M records):
   the index now keeps keys up to `max_df` (default 10,000). Each S1 record queries
   with its RAREST keys first, adding keys while the summed posting-list length stays
   within `budget`, and always uses at least its single rarest key. Common-word names
   still get candidates. Chunks are sized by total retrieval COST (not record count),
   so memory stays bounded however skewed the data is.
3. Two-stage scoring: stage 1 (budgeted keys) keeps K_pre candidates; stage 2
   recomputes the FULL IDF overlap over all of the record's keys for those candidates
   and re-ranks to the final K. Better ranking means higher recall at the same K.
4. 43 features for LightGBM: postal-code / house-number agreement and conflict, legal
   form, acronyms, first token, raw-name similarity, record information content, etc.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from normalize import normalize
from spell import build_corrections

TYP_W = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}

FEATURES = [
    # blocking / IDF overlap
    "score_l", "s_name_l", "s_addr_l", "s_dig", "s1score_l",
    "name_cov1", "name_cov2", "addr_cov1", "addr_cov2", "tn1_l", "tn2_l", "ta1_l", "ta2_l",
    "rank", "score_rel", "score_gap", "n_cand",
    # numbers
    "dig_both", "dig_conflict", "pc_eq", "pc3_eq", "pc_conflict", "pc_missing", "hn_eq", "hn_conflict",
    "fz_dig_tset",
    # names
    "fz_n_tset", "fz_n_tsort", "fz_n_ratio", "fz_n_jw", "fz_n_partial", "fz_nraw_ratio",
    "first_eq", "acro", "legal_eq", "legal_conflict", "ntok_ratio", "nk1",
    # addresses
    "fz_a_tsort", "fz_a_tset", "fz_a_partial", "atok_ratio",
    # source
    "is_s3",
]

KEEP_COLS = ["idx", "entity_id", "cty", "name_key", "addr_key", "name_raw", "name_ntok", "addr_ntok",
             "pc", "pc3", "hn", "dig_key", "legal", "initials", "first_tok"]


def _keys(nd, fix, batch=1_000_000):
    parts = []
    for off in range(0, nd.height, batch):
        b = nd.slice(off, batch)
        n = b.select("idx", "cty", pl.col("name_toks").alias("tok")).explode("tok").drop_nulls("tok")
        a = b.select("idx", "cty", pl.col("addr_toks").alias("tok")).explode("tok").drop_nulls("tok")
        if fix is not None and fix.height:
            n = n.join(fix, on=["cty", "tok"], how="left").with_columns(pl.coalesce("fix", "tok").alias("tok")).drop("fix")
            a = a.join(fix, on=["cty", "tok"], how="left").with_columns(pl.coalesce("fix", "tok").alias("tok")).drop("fix")
        n = n.with_columns(pl.col("tok").shift(-1).over("idx").alias("nxt"))
        uni = n.select("idx", "cty", pl.lit(0, pl.UInt8).alias("typ"), "tok")
        bi = n.filter(pl.col("nxt").is_not_null()).select(
            "idx", "cty", pl.lit(1, pl.UInt8).alias("typ"), (pl.col("tok") + "_" + pl.col("nxt")).alias("tok"))
        has_d = pl.col("tok").str.contains(r"\d")
        a = a.filter(has_d | (pl.col("tok").str.len_chars() >= 2)).select(
            "idx", "cty", pl.when(has_d).then(pl.lit(3, pl.UInt8)).otherwise(pl.lit(2, pl.UInt8)).alias("typ"), "tok")
        parts.append(pl.concat([uni, bi, a]).select(
            "idx", "typ",
            pl.concat_str([pl.col("cty"), pl.col("typ").cast(pl.Utf8), pl.col("tok")], separator="|")
            .hash(seed=42).alias("key"),
        ).unique())
    return pl.concat(parts)


def _rec_stats(kw, k_all, n):
    s = kw.group_by("idx").agg(
        pl.col("w").filter(pl.col("typ") <= 1).sum().alias("tot_name"),
        pl.col("w").filter(pl.col("typ") == 2).sum().alias("tot_addr"),
    )
    d = k_all.filter(pl.col("typ") == 3).group_by("idx").agg(pl.len().cast(pl.Float32).alias("n_dig"))
    base = pl.DataFrame({"idx": pl.arange(0, n, eager=True).cast(pl.UInt32)})
    return base.join(s, on="idx", how="left").join(d, on="idx", how="left").fill_null(0).sort("idx")


class Corpus2:
    def __init__(self, s1_raw, tgt_raw, max_df=10_000, budget=3_000, spell=True, log=print):
        s1, tg = normalize(s1_raw), normalize(tgt_raw)
        fix = build_corrections([s1, tg]) if spell else None
        log(f"  spelling corrections: {0 if fix is None else fix.height:,}")
        k1, kt = _keys(s1, fix), _keys(tg, fix)
        self.s1, self.tg = s1.select(KEEP_COLS), tg.select(KEEP_COLS)
        del s1, tg
        dft = kt.group_by("key").agg(pl.len().cast(pl.UInt32).alias("dft"))
        nt = self.tg.height

        def weigh(k):
            return (k.join(dft, on="key", how="left").with_columns(pl.col("dft").fill_null(0))
                    .filter(pl.col("dft") <= max_df)
                    .with_columns((((nt + 1) / (pl.col("dft") + 1)).log()
                                   * pl.col("typ").replace_strict(TYP_W, return_dtype=pl.Float64))
                                  .cast(pl.Float32).alias("w")))

        k1w, ktw = weigh(k1), weigh(kt)
        self.s1_stats = _rec_stats(k1w, k1, self.s1.height)
        self.tg_stats = _rec_stats(ktw, kt, self.tg.height)
        del k1, kt
        # S1 keys sorted by (idx, rarity) -> budget selection is a cumulative sum
        k1s = k1w.filter(pl.col("dft") > 0).select("idx", "typ", "key", "w", "dft").sort(["idx", "dft", "key"])
        self.k1 = k1s.with_columns(
            ((pl.col("dft").cast(pl.Int64).cum_sum().over("idx") <= budget)
             | (pl.int_range(pl.len()).over("idx") == 0)).alias("sel"))
        cost = self.k1.filter("sel").group_by("idx").agg(pl.col("dft").cast(pl.Int64).sum().alias("c"))
        c = np.zeros(self.s1.height, dtype=np.int64)
        c[cost["idx"].to_numpy()] = cost["c"].to_numpy()
        self._cost_cum = np.cumsum(c)
        self.kt = ktw.select("key", pl.col("idx").alias("idx2"))
        self._k1_idx = self.k1["idx"]
        self.budget = budget
        self.is_s3 = self.tg["entity_id"].str.starts_with("S3-").cast(pl.Float32)

    # ------------------------------------------------------------------ blocking
    def candidates(self, lo, hi, K, K_pre):
        a = int(self._k1_idx.search_sorted(lo, side="left"))
        b = int(self._k1_idx.search_sorted(hi, side="left"))
        q = self.k1.slice(a, b - a)
        if q.height == 0:
            return None
        sel = q.filter("sel")
        # stage 1: retrieval with budgeted rare keys
        c = sel.select("idx", "key", "w").join(self.kt, on="key").group_by("idx", "idx2").agg(
            pl.col("w").sum().alias("s1score"))
        c = c.sort(["idx", "s1score", "idx2"], descending=[False, True, False]) \
             .filter(pl.int_range(pl.len()).over("idx") < K_pre)
        if c.height == 0:
            return None
        # stage 2: full overlap over ALL of the record's keys, restricted to retrieved targets
        tk = self.kt.filter(pl.col("idx2").is_in(c["idx2"].unique().implode()))
        full = c.select("idx", "idx2").join(q.select("idx", "key", "w", "typ"), on="idx") \
                .join(tk, on=["idx2", "key"], how="semi")
        agg = full.group_by("idx", "idx2").agg(
            pl.col("w").sum().alias("score"),
            pl.col("w").filter(pl.col("typ") <= 1).sum().alias("s_name"),
            pl.col("w").filter(pl.col("typ") == 2).sum().alias("s_addr"),
            (pl.col("typ") == 3).sum().cast(pl.Float32).alias("s_dig"),
        )
        c = c.join(agg, on=["idx", "idx2"], how="left").fill_null(0)
        c = c.sort(["idx", "score", "s1score", "idx2"], descending=[False, True, True, False]) \
             .with_columns(pl.int_range(pl.len()).over("idx").alias("rank")).filter(pl.col("rank") < K)
        return c

    # ------------------------------------------------------------------ features
    def features(self, c):
        i1, i2 = c["idx"], c["idx2"]
        S, T, A, B = self.s1_stats, self.tg_stats, self.s1, self.tg

        def g(df, col, idx, name):
            return df[col].gather(idx).alias(name)

        def div(a, b):
            return pl.when(b > 0).then(a / b).otherwise(0.0)

        both = lambda x, y: pl.col(x).is_not_null() & pl.col(y).is_not_null() & (pl.col(x) != "") & (pl.col(y) != "")
        f = c.with_columns(
            g(S, "tot_name", i1, "tn1"), g(T, "tot_name", i2, "tn2"),
            g(S, "tot_addr", i1, "ta1"), g(T, "tot_addr", i2, "ta2"),
            g(S, "n_dig", i1, "nd1"), g(T, "n_dig", i2, "nd2"),
            g(A, "name_ntok", i1, "nk1"), g(B, "name_ntok", i2, "nk2"),
            g(A, "addr_ntok", i1, "ak1"), g(B, "addr_ntok", i2, "ak2"),
            g(A, "pc", i1, "pc1"), g(B, "pc", i2, "pc2"), g(A, "pc3", i1, "p31"), g(B, "pc3", i2, "p32"),
            g(A, "hn", i1, "hn1"), g(B, "hn", i2, "hn2"),
            g(A, "legal", i1, "lg1"), g(B, "legal", i2, "lg2"),
            g(A, "initials", i1, "in1"), g(B, "initials", i2, "in2"),
            g(A, "name_key", i1, "nm1"), g(B, "name_key", i2, "nm2"),
            g(A, "first_tok", i1, "ft1"), g(B, "first_tok", i2, "ft2"),
            self.is_s3.gather(i2).alias("is_s3"),
        ).with_columns(
            pl.col("score").log1p().alias("score_l"), pl.col("s_name").log1p().alias("s_name_l"),
            pl.col("s_addr").log1p().alias("s_addr_l"), pl.col("s1score").log1p().alias("s1score_l"),
            div(pl.col("s_name"), pl.col("tn1")).alias("name_cov1"), div(pl.col("s_name"), pl.col("tn2")).alias("name_cov2"),
            div(pl.col("s_addr"), pl.col("ta1")).alias("addr_cov1"), div(pl.col("s_addr"), pl.col("ta2")).alias("addr_cov2"),
            pl.col("tn1").log1p().alias("tn1_l"), pl.col("tn2").log1p().alias("tn2_l"),
            pl.col("ta1").log1p().alias("ta1_l"), pl.col("ta2").log1p().alias("ta2_l"),
            pl.col("rank").cast(pl.Float32),
            (pl.col("score") / pl.col("score").max().over("idx")).alias("score_rel"),
            (pl.col("score").max().over("idx") - pl.col("score")).alias("score_gap"),
            pl.len().over("idx").cast(pl.Float32).alias("n_cand"),
            ((pl.col("nd1") > 0) & (pl.col("nd2") > 0)).cast(pl.Float32).alias("dig_both"),
            ((pl.col("nd1") > 0) & (pl.col("nd2") > 0) & (pl.col("s_dig") == 0)).cast(pl.Float32).alias("dig_conflict"),
            (both("pc1", "pc2") & (pl.col("pc1") == pl.col("pc2"))).cast(pl.Float32).alias("pc_eq"),
            (both("p31", "p32") & (pl.col("p31") == pl.col("p32"))).cast(pl.Float32).alias("pc3_eq"),
            (both("pc1", "pc2") & (pl.col("pc1") != pl.col("pc2"))).cast(pl.Float32).alias("pc_conflict"),
            (~both("pc1", "pc2")).cast(pl.Float32).alias("pc_missing"),
            (both("hn1", "hn2") & (pl.col("hn1") == pl.col("hn2"))).cast(pl.Float32).alias("hn_eq"),
            (both("hn1", "hn2") & (pl.col("hn1") != pl.col("hn2"))).cast(pl.Float32).alias("hn_conflict"),
            (both("lg1", "lg2") & (pl.col("lg1") == pl.col("lg2"))).cast(pl.Float32).alias("legal_eq"),
            (both("lg1", "lg2") & (pl.col("lg1") != pl.col("lg2"))).cast(pl.Float32).alias("legal_conflict"),
            (both("ft1", "ft2") & (pl.col("ft1") == pl.col("ft2"))).cast(pl.Float32).alias("first_eq"),
            (((pl.col("in1") == pl.col("nm2")) & (pl.col("nk2") == 1) & (pl.col("nk1") >= 2))
             | ((pl.col("in2") == pl.col("nm1")) & (pl.col("nk1") == 1) & (pl.col("nk2") >= 2))
             | (both("in1", "in2") & (pl.col("in1") == pl.col("in2")) & (pl.col("nk1") >= 2) & (pl.col("nk2") >= 2)))
            .cast(pl.Float32).alias("acro"),
            div(pl.min_horizontal("nk1", "nk2"), pl.max_horizontal("nk1", "nk2")).alias("ntok_ratio"),
            div(pl.min_horizontal("ak1", "ak2"), pl.max_horizontal("ak1", "ak2")).alias("atok_ratio"),
        )
        L = lambda df, col, idx: df[col].gather(idx).fill_null("").to_list()
        n1, n2 = L(A, "name_key", i1), L(B, "name_key", i2)
        r1, r2 = L(A, "name_raw", i1), L(B, "name_raw", i2)
        a1, a2 = L(A, "addr_key", i1), L(B, "addr_key", i2)
        d1, d2 = L(A, "dig_key", i1), L(B, "dig_key", i2)
        kw = dict(workers=-1, dtype=np.float32)
        cp = lambda x, y, s: process.cpdist(x, y, scorer=s, **kw)
        f = f.with_columns(
            pl.Series("fz_n_tset", cp(n1, n2, fuzz.token_set_ratio)),
            pl.Series("fz_n_tsort", cp(n1, n2, fuzz.token_sort_ratio)),
            pl.Series("fz_n_ratio", cp(n1, n2, fuzz.ratio)),
            pl.Series("fz_n_jw", cp(n1, n2, JaroWinkler.normalized_similarity) * 100),
            pl.Series("fz_n_partial", cp(n1, n2, fuzz.partial_ratio)),
            pl.Series("fz_nraw_ratio", cp(r1, r2, fuzz.ratio)),
            pl.Series("fz_a_tsort", cp(a1, a2, fuzz.token_sort_ratio)),
            pl.Series("fz_a_tset", cp(a1, a2, fuzz.token_set_ratio)),
            pl.Series("fz_a_partial", cp(a1, a2, fuzz.partial_ratio)),
            pl.Series("fz_dig_tset", cp(d1, d2, fuzz.token_set_ratio)),
        )
        return f.select("idx", "idx2", *FEATURES)

    def chunk_bounds(self, chunk, cost_cap):
        """Record ranges whose total retrieval cost <= cost_cap (and <= `chunk` records)."""
        n, cc, lo = self.s1.height, self._cost_cum, 0
        while lo < n:
            base = cc[lo - 1] if lo else 0
            hi = int(np.searchsorted(cc, base + cost_cap, side="right"))
            hi = max(lo + 1, min(hi, lo + chunk, n))
            yield lo, hi
            lo = hi

    def iter_chunks(self, K, K_pre, chunk, cost_cap=20_000_000):
        for lo, hi in self.chunk_bounds(chunk, cost_cap):
            c = self.candidates(lo, hi, K, K_pre)
            if c is not None and c.height:
                yield self.features(c)


def X_of(f):
    return f.select(FEATURES).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
