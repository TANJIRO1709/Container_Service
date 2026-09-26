"""Phase 1 core: IDF-weighted inverted-index blocking + pairwise features.

Blocking idea
-------------
Every record is turned into hashed "keys" scoped by country:
  typ 0  name unigram        (weight 1.0)
  typ 1  name adjacent bigram (weight 1.0)  -> rarer, rescues common-word names
  typ 2  address word        (weight 0.5)
  typ 3  address token with a digit: house no., PIN/ZIP/postcode (weight 1.0)
Key weight = IDF on the target corpus (S2+S3). Keys appearing in more than
`max_df` target records are too common to block on and are dropped, which bounds
the join size. Candidate score = sum of weights of shared keys; keep top-K per S1.

Everything is vectorised (polars joins + rapidfuzz C++ batch scorers) and processed
in S1 chunks so memory stays bounded on millions of records.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from normalize import normalize

TYP_W = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}

FEATURES = [
    "score_l", "s_name_l", "s_addr_l", "s_dig",
    "name_cov1", "name_cov2", "addr_cov1", "addr_cov2",
    "dig_both", "dig_conflict",
    "rank", "score_rel", "n_cand",
    "fz_n_tset", "fz_n_ratio", "fz_n_jw", "fz_a_tsort", "fz_a_tset",
    "ntok_ratio", "is_s3",
]


def _keys(nd: pl.DataFrame, batch: int = 1_000_000) -> pl.DataFrame:
    parts = []
    for off in range(0, nd.height, batch):
        b = nd.slice(off, batch)
        n = b.select("idx", "cty", pl.col("name_toks").alias("tok")).explode("tok").drop_nulls("tok")
        n = n.with_columns(pl.col("tok").shift(-1).over("idx").alias("nxt"))
        uni = n.select("idx", "cty", pl.lit(0, pl.UInt8).alias("typ"), "tok")
        bi = n.filter(pl.col("nxt").is_not_null()).select(
            "idx", "cty", pl.lit(1, pl.UInt8).alias("typ"), (pl.col("tok") + "_" + pl.col("nxt")).alias("tok"))
        a = b.select("idx", "cty", pl.col("addr_toks").alias("tok")).explode("tok").drop_nulls("tok")
        has_d = pl.col("tok").str.contains(r"\d")
        a = a.filter(has_d | (pl.col("tok").str.len_chars() >= 2)).select(
            "idx", "cty", pl.when(has_d).then(pl.lit(3, pl.UInt8)).otherwise(pl.lit(2, pl.UInt8)).alias("typ"), "tok")
        k = pl.concat([uni, bi, a]).select(
            "idx", "typ",
            pl.concat_str([pl.col("cty"), pl.col("typ").cast(pl.Utf8), pl.col("tok")], separator="|")
            .hash(seed=42).alias("key"),
        ).unique()
        parts.append(k)
    return pl.concat(parts)


def _rec_stats(kw: pl.DataFrame, k_all: pl.DataFrame, n: int) -> pl.DataFrame:
    s = kw.group_by("idx").agg(
        pl.col("w").filter(pl.col("typ") <= 1).sum().alias("tot_name"),
        pl.col("w").filter(pl.col("typ") == 2).sum().alias("tot_addr"),
    )
    d = k_all.filter(pl.col("typ") == 3).group_by("idx").agg(pl.len().cast(pl.Float32).alias("n_dig"))
    base = pl.DataFrame({"idx": pl.arange(0, n, eager=True).cast(pl.UInt32)})
    return base.join(s, on="idx", how="left").join(d, on="idx", how="left").fill_null(0).sort("idx")


class Corpus:
    """Holds normalized S1 + target (S2+S3) records and the blocking index."""

    def __init__(self, s1_raw: pl.DataFrame, tgt_raw: pl.DataFrame, max_df: int = 500):
        self.s1 = normalize(s1_raw)
        self.tg = normalize(tgt_raw)
        k1, kt = _keys(self.s1), _keys(self.tg)
        dft = kt.group_by("key").agg(pl.len().cast(pl.UInt32).alias("dft"))
        nt = self.tg.height

        def weigh(k):
            return (
                k.join(dft, on="key", how="left").with_columns(pl.col("dft").fill_null(0))
                .filter(pl.col("dft") <= max_df)
                .with_columns(
                    (((nt + 1) / (pl.col("dft") + 1)).log()
                     * pl.col("typ").replace_strict(TYP_W, return_dtype=pl.Float64)).cast(pl.Float32).alias("w"))
            )

        k1w, ktw = weigh(k1), weigh(kt)
        self.s1_stats = _rec_stats(k1w, k1, self.s1.height)
        self.tg_stats = _rec_stats(ktw, kt, self.tg.height)
        self.k1 = k1w.filter(pl.col("dft") > 0).select("idx", "typ", "key", "w").sort("idx")
        self.kt = ktw.select("key", pl.col("idx").alias("idx2"))
        self.s1 = self.s1.drop("name_toks", "addr_toks")
        self.tg = self.tg.drop("name_toks", "addr_toks")
        self.is_s3 = self.tg["entity_id"].str.starts_with("S3-").cast(pl.Float32)
        self._k1_idx = self.k1["idx"]

    # ---------------------------------------------------------------- blocking
    def candidates(self, lo: int, hi: int, K: int) -> pl.DataFrame:
        a = int(self._k1_idx.search_sorted(lo, side="left"))
        b = int(self._k1_idx.search_sorted(hi, side="left"))
        q = self.k1.slice(a, b - a)
        agg = q.join(self.kt, on="key").group_by("idx", "idx2").agg(
            pl.col("w").sum().alias("score"),
            pl.col("w").filter(pl.col("typ") <= 1).sum().alias("s_name"),
            pl.col("w").filter(pl.col("typ") == 2).sum().alias("s_addr"),
            (pl.col("typ") == 3).sum().cast(pl.Float32).alias("s_dig"),
        )
        agg = agg.sort(["idx", "score", "idx2"], descending=[False, True, False])
        return agg.with_columns(pl.int_range(pl.len()).over("idx").alias("rank")).filter(pl.col("rank") < K)

    # ---------------------------------------------------------------- features
    def features(self, c: pl.DataFrame) -> pl.DataFrame:
        if c.height == 0:
            return c
        i1, i2 = c["idx"], c["idx2"]
        S, T = self.s1_stats, self.tg_stats

        def div(a, b):
            return pl.when(b > 0).then(a / b).otherwise(0.0)

        f = c.with_columns(
            S["tot_name"].gather(i1).alias("tn1"), T["tot_name"].gather(i2).alias("tn2"),
            S["tot_addr"].gather(i1).alias("ta1"), T["tot_addr"].gather(i2).alias("ta2"),
            S["n_dig"].gather(i1).alias("nd1"), T["n_dig"].gather(i2).alias("nd2"),
            self.s1["name_ntok"].gather(i1).alias("nk1"), self.tg["name_ntok"].gather(i2).alias("nk2"),
            self.is_s3.gather(i2).alias("is_s3"),
        ).with_columns(
            pl.col("score").log1p().alias("score_l"),
            pl.col("s_name").log1p().alias("s_name_l"),
            pl.col("s_addr").log1p().alias("s_addr_l"),
            div(pl.col("s_name"), pl.col("tn1")).alias("name_cov1"),
            div(pl.col("s_name"), pl.col("tn2")).alias("name_cov2"),
            div(pl.col("s_addr"), pl.col("ta1")).alias("addr_cov1"),
            div(pl.col("s_addr"), pl.col("ta2")).alias("addr_cov2"),
            ((pl.col("nd1") > 0) & (pl.col("nd2") > 0)).cast(pl.Float32).alias("dig_both"),
            ((pl.col("nd1") > 0) & (pl.col("nd2") > 0) & (pl.col("s_dig") == 0)).cast(pl.Float32).alias("dig_conflict"),
            (pl.col("score") / pl.col("score").max().over("idx")).alias("score_rel"),
            pl.len().over("idx").cast(pl.Float32).alias("n_cand"),
            div(pl.min_horizontal("nk1", "nk2"), pl.max_horizontal("nk1", "nk2")).alias("ntok_ratio"),
            pl.col("rank").cast(pl.Float32),
        )
        n1 = self.s1["name_key"].gather(i1).to_list()
        n2 = self.tg["name_key"].gather(i2).to_list()
        a1 = self.s1["addr_key"].gather(i1).to_list()
        a2 = self.tg["addr_key"].gather(i2).to_list()
        kw = dict(workers=-1, dtype=np.float32)
        f = f.with_columns(
            pl.Series("fz_n_tset", process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, **kw)),
            pl.Series("fz_n_ratio", process.cpdist(n1, n2, scorer=fuzz.ratio, **kw)),
            pl.Series("fz_n_jw", process.cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, **kw) * 100),
            pl.Series("fz_a_tsort", process.cpdist(a1, a2, scorer=fuzz.token_sort_ratio, **kw)),
            pl.Series("fz_a_tset", process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, **kw)),
        )
        return f.select("idx", "idx2", *FEATURES)

    def iter_chunks(self, K: int, chunk: int):
        for lo in range(0, self.s1.height, chunk):
            yield self.features(self.candidates(lo, lo + chunk, K))


def X_of(f: pl.DataFrame) -> np.ndarray:
    return f.select(FEATURES).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
