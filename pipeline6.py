"""Phase 6 core: Phase 3 + a character n-gram retrieval channel (channel C).

Word keys — even with typo correction, phonetics and acronyms — need SOME shared token.
Character n-grams need none: "Vishwakarma Engg" and "Viswakarrma Engineering" share the
5-grams "iswak"/"swaka" region regardless of how the words are split or misspelt. This is
the CPU-feasible stand-in for dense-embedding retrieval: same goal (similarity without
shared vocabulary), no GPU and no model.

Two tricks keep it affordable at 10M+ records:
  * HASH SAMPLING, not striding. Every n-gram is kept iff hash(gram) %% ngram_mod == 0.
    Sampling by position would break as soon as one character is inserted or deleted;
    sampling by the gram's own hash keeps the SAME grams in every record, so two similar
    strings still share the sampled ones.
  * The keys join channel C, which can only ADD candidates on top of channel A, never
    displace them (the Phase-3 lesson).

---- Phase 3 notes below ----
Phase 3 core: Phase 2 + four NEW retrieval channels aimed at matches that share
no plain word key (Phase 2 left ~15% of true pairs unretrievable on real data).

New key types (all country-scoped, IDF-weighted, budget-selected like the rest):
  typ 4  phonetic/transliteration skeleton of name tokens   lakshmi / laxmi -> "lksm"
  typ 5  acronym key                                        "State Bank of India" <-> "SBI"
  typ 6  house-number | postal-code composite               "12|751001" (very rare = very strong)
  typ 7  house-number | street-name composite               "12|mahatma"
  typ 8  name-token | postal-prefix composite               "traders|751" (rescues common-word names)
  typ 9  joined tokens: adjacent name tokens concatenated      "sun rise" <-> "sunrise"
Channel A (typ 0-3) is now byte-for-byte the Phase-2 key set.
Typ 6/7 make DBA/trade-name matches (different name, same address) retrievable.

UNION OF BLOCKERS: channel A (Phase-2 keys, typ 0-3) and channel B (new keys, typ 4-8)
retrieve and rank INDEPENDENTLY; final candidates = top-K of A  U  top-K_b of B.
(Merging them into one score let composite keys crowd out true matches whose postal
code is missing, so B must only ever ADD candidates, never displace A's.)

---- Phase 2 notes below ----
Phase 2 core.

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

TYP_W = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0, 4: 0.7, 5: 0.6, 6: 1.5, 7: 1.0, 8: 0.8, 9: 0.8,
         10: 0.9, 11: 0.5}

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
    # phase 3 channels
    "s_phon_l", "s_comp_l", "hnpc_shared", "rank_b", "s1b_l",
    # phase 6 character n-grams
    "s_ngn_l", "s_nga_l", "ng_cov1", "ng_cov2", "rank_c", "s1c_l",
]

# generic address words never used as the "street name" in typ-7 keys
ADDR_GENERIC = ["near", "opposite", "plot", "no", "number", "flat", "shop", "floor", "building", "block",
                "sector", "road", "street", "avenue", "lane", "nagar", "colony", "main", "cross", "rue",
                "de", "du", "la", "le", "des", "boulevard", "chemin", "suite", "apartment", "po", "ps",
                "dist", "district", "behind", "beside", "next", "to", "the", "and", "of", "unit", "house"]


def phonetic(e: pl.Expr) -> pl.Expr:
    """Transliteration-tolerant consonant skeleton (Indic/English/French friendly)."""
    e = (e.str.replace_all("ksh", "ks").str.replace_all("x", "ks").str.replace_all("ph", "f")
         .str.replace_all("bh", "b").str.replace_all("dh", "d").str.replace_all("th", "t")
         .str.replace_all("kh", "k").str.replace_all("gh", "g").str.replace_all("sh", "s")
         .str.replace_all("ch", "c").str.replace_all("ck", "k").str.replace_all("c", "k")
         .str.replace_all("q", "k").str.replace_all("w", "v").str.replace_all("z", "j"))
    first = e.str.slice(0, 1)
    rest = e.str.slice(1).str.replace_all(r"[aeiouy]", "")
    out = first + rest
    for ch in "bdfgjklmnprstv":            # collapse doubled consonants (no regex backrefs in Rust)
        out = out.str.replace_all(ch + "+", ch)
    return out

KEEP_COLS = ["idx", "entity_id", "cty", "name_key", "addr_key", "name_raw", "name_ntok", "addr_ntok",
             "pc", "pc3", "hn", "dig_key", "legal", "initials", "first_tok"]


NGRAM_N = 5          # character n-gram length


def _keys(nd, fix, batch=750_000, ngram_mod=4):
    parts = []
    for off in range(0, nd.height, batch):
        b = nd.slice(off, batch)
        n = b.select("idx", "cty", pl.col("name_toks").alias("tok")).explode("tok").drop_nulls("tok")
        a = b.select("idx", "cty", pl.col("addr_toks").alias("tok")).explode("tok").drop_nulls("tok")
        if fix is not None and fix.height:
            n = n.join(fix, on=["cty", "tok"], how="left").with_columns(pl.coalesce("fix", "tok").alias("tok")).drop("fix")
            a = a.join(fix, on=["cty", "tok"], how="left").with_columns(pl.coalesce("fix", "tok").alias("tok")).drop("fix")
        n = n.with_columns(pl.col("tok").shift(-1).over("idx").alias("nxt"))
        a = a.with_columns(pl.int_range(pl.len()).over("idx").alias("pos"))
        rec = b.select("idx", "cty", "hn", "pc", "pc3", pl.col("name_toks").list.len().alias("nk"),
                       pl.col("name_toks").list.first().alias("t0"))
        L = lambda t: pl.lit(t, pl.UInt8).alias("typ")
        alpha = pl.col("tok").str.contains(r"^\p{L}+$")

        uni = n.select("idx", "cty", L(0), "tok")
        pairs_ = n.filter(pl.col("nxt").is_not_null())
        bi = pairs_.select("idx", "cty", L(1), (pl.col("tok") + "_" + pl.col("nxt")).alias("tok"))
        # typ 9 (channel B): joined adjacent tokens "sun rise" -> "sunrise", matched against both
        # other records' joined forms (typ 9) and their plain unigrams (emitted again as typ 9 below)
        comp = pairs_.select("idx", "cty", L(9), (pl.col("tok") + pl.col("nxt")).alias("tok"))
        uni9 = n.filter(pl.col("tok").str.len_chars() >= 6).select("idx", "cty", L(9), "tok")
        has_d = pl.col("tok").str.contains(r"\d")
        ad = a.filter(has_d | (pl.col("tok").str.len_chars() >= 2)).select(
            "idx", "cty", pl.when(has_d).then(pl.lit(3, pl.UInt8)).otherwise(pl.lit(2, pl.UInt8)).alias("typ"), "tok")
        # typ 4: phonetic skeleton of alphabetic name tokens
        ph = n.filter(alpha & (pl.col("tok").str.len_chars() >= 4)).select(
            "idx", "cty", L(4), phonetic(pl.col("tok")).alias("tok")).filter(pl.col("tok").str.len_chars() >= 3)
        # typ 5: acronym (initials of multi-token names; short single-token names as-is)
        ini = n.group_by("idx", maintain_order=True).agg(pl.col("tok").str.slice(0, 1).str.join("").alias("ini"))
        acr = rec.join(ini, on="idx", how="left").select(
            "idx", "cty", L(5),
            pl.when(pl.col("nk") >= 2).then(pl.col("ini"))
            .when((pl.col("nk") == 1) & pl.col("t0").str.contains(r"^\p{L}{2,5}$")).then(pl.col("t0"))
            .otherwise(None).alias("tok")).drop_nulls("tok").filter(pl.col("tok").str.len_chars() >= 2)
        # typ 6: house number | postal code
        hp = rec.filter(pl.col("hn").is_not_null() & pl.col("pc").is_not_null()).select(
            "idx", "cty", L(6), (pl.col("hn") + "|" + pl.col("pc")).alias("tok"))
        # typ 7: house number | first non-generic street word after it
        hpos = a.join(rec.select("idx", "hn"), on="idx").filter(pl.col("tok") == pl.col("hn")) \
                .group_by("idx").agg(pl.col("pos").min().alias("hpos"), pl.col("hn").first())
        street = a.join(hpos, on="idx").filter(
            (pl.col("pos") > pl.col("hpos")) & alpha & (pl.col("tok").str.len_chars() >= 3)
            & ~pl.col("tok").is_in(ADDR_GENERIC)
        ).sort(["idx", "pos"]).group_by("idx", maintain_order=True).first()
        hs = street.select("idx", "cty", L(7), (pl.col("hn") + "|" + pl.col("tok")).alias("tok"))
        # typ 8: name token | postal prefix
        npc = n.join(rec.select("idx", "pc3").drop_nulls("pc3"), on="idx").select(
            "idx", "cty", L(8), (pl.col("tok") + "|" + pl.col("pc3")).alias("tok"))

        # typ 10/11 (channel C): hash-sampled character n-grams of the name / address
        def grams(src_col, typ, n=NGRAM_N):
            t = b.select("idx", "cty", pl.col(src_col).str.replace_all(" ", "").alias("s")) \
                 .filter(pl.col("s").str.len_chars() >= n)
            t = t.with_columns(pl.int_ranges(0, pl.col("s").str.len_chars() - n + 1).alias("o")).explode("o")
            return (t.with_columns(pl.col("s").str.slice(pl.col("o"), n).alias("tok"))
                    .filter(pl.col("tok").hash(seed=99) % ngram_mod == 0)
                    .select("idx", "cty", pl.lit(typ, pl.UInt8).alias("typ"), "tok"))

        chan_c = [grams("name_key", 10), grams("addr_key", 11)] if ngram_mod else []
        allk = pl.concat([uni, bi, comp, uni9, ad, ph, acr, hp, hs, npc] + chan_c, how="vertical_relaxed")
        parts.append(allk.select(
            "idx", "typ",
            pl.concat_str([pl.col("cty"), pl.col("typ").cast(pl.Utf8), pl.col("tok")], separator="|")
            .hash(seed=42).alias("key"),
        ).unique())
    return pl.concat(parts)


def _rec_stats(kw, k_all, n):
    s = kw.group_by("idx").agg(
        pl.col("w").filter(pl.col("typ") <= 1).sum().alias("tot_name"),
        pl.col("w").filter(pl.col("typ") == 2).sum().alias("tot_addr"),
        pl.col("w").filter(pl.col("typ") == 10).sum().alias("tot_ng"),
    )
    d = k_all.filter(pl.col("typ") == 3).group_by("idx").agg(pl.len().cast(pl.Float32).alias("n_dig"))
    base = pl.DataFrame({"idx": pl.arange(0, n, eager=True).cast(pl.UInt32)})
    return base.join(s, on="idx", how="left").join(d, on="idx", how="left").fill_null(0).sort("idx")


class Corpus6:
    def __init__(self, s1_raw, tgt_raw, max_df=10_000, budget=3_000, spell=True, log=print, batch=750_000,
                 ngram_mod=4):
        s1, tg = normalize(s1_raw, batch), normalize(tgt_raw, batch)
        fix = build_corrections([s1, tg], batch=batch) if spell else None
        log(f"  spelling corrections: {0 if fix is None else fix.height:,}")
        k1, kt = _keys(s1, fix, batch, ngram_mod), _keys(tg, fix, batch, ngram_mod)
        self.s1, self.tg = s1.select(KEEP_COLS), tg.select(KEEP_COLS)
        del s1, tg
        dft = kt.group_by("key").agg(pl.len().cast(pl.UInt32).alias("dft"))
        nt = self.tg.height

        def weigh(k, wbatch=20_000_000):
            outs = []
            for o in range(0, k.height, wbatch):   # batched: this join otherwise doubles peak memory
                outs.append(
                    k.slice(o, wbatch).join(dft, on="key", how="left").with_columns(pl.col("dft").fill_null(0))
                    .filter(pl.col("dft") <= max_df)
                    .with_columns((((nt + 1) / (pl.col("dft") + 1)).log()
                                   * pl.col("typ").replace_strict(TYP_W, return_dtype=pl.Float64))
                                  .cast(pl.Float32).alias("w")))
            return pl.concat(outs, rechunk=True) if len(outs) > 1 else outs[0]

        k1w, ktw = weigh(k1), weigh(kt)
        self.s1_stats = _rec_stats(k1w, k1, self.s1.height)
        self.tg_stats = _rec_stats(ktw, kt, self.tg.height)
        del k1, kt
        # S1 keys sorted by (idx, rarity) -> budget selection is a cumulative sum
        k1s = k1w.filter(pl.col("dft") > 0).select("idx", "typ", "key", "w", "dft").sort(["idx", "dft", "key"])
        grp = ["idx", "ch"]
        self.k1 = k1s.with_columns(pl.when(pl.col("typ") <= 3).then(0)
                                   .when(pl.col("typ") <= 9).then(1).otherwise(2)
                                   .cast(pl.UInt8).alias("ch")) \
            .with_columns(((pl.col("dft").cast(pl.Int64).cum_sum().over(grp) <= budget)
                           | (pl.int_range(pl.len()).over(grp) == 0)).alias("sel"))
        cost = self.k1.filter("sel").group_by("idx").agg(pl.col("dft").cast(pl.Int64).sum().alias("c"))
        c = np.zeros(self.s1.height, dtype=np.int64)
        c[cost["idx"].to_numpy()] = cost["c"].to_numpy()
        self._cost_cum = np.cumsum(c)
        self.kt = ktw.select("key", pl.col("idx").alias("idx2"))
        self._k1_idx = self.k1["idx"]
        self.budget = budget
        self.is_s3 = self.tg["entity_id"].str.starts_with("S3-").cast(pl.Float32)

    # ------------------------------------------------------------------ blocking
    def candidates(self, lo, hi, K, K_pre, K_b=20, K_pre_b=50, K_c=10, K_pre_c=30):
        a = int(self._k1_idx.search_sorted(lo, side="left"))
        b = int(self._k1_idx.search_sorted(hi, side="left"))
        q = self.k1.slice(a, b - a)
        if q.height == 0:
            return None
        hits = q.filter("sel").select("idx", "key", "w", "ch").join(self.kt, on="key")

        # Parallel float sums are order-dependent, so candidates tied at the top-K cutoff
        # could swap between runs. Rounding makes ties exact; idx2 then breaks them.
        R = lambda e: e.round(4)

        def top(ch, n, name):
            t = hits.filter(pl.col("ch") == ch).group_by("idx", "idx2").agg(R(pl.col("w").sum()).alias(name))
            return t.sort(["idx", name, "idx2"], descending=[False, True, False]) \
                    .filter(pl.int_range(pl.len()).over("idx") < n)

        c = top(0, K_pre, "s1score")
        for ch, n, nm in ((1, K_pre_b, "s1b"), (2, K_pre_c, "s1c")):
            c = c.join(top(ch, n, nm), on=["idx", "idx2"], how="full", coalesce=True)
        c = c.fill_null(0)
        if c.height == 0:
            return None
        # stage 2: full IDF overlap over ALL of the record's keys, restricted to retrieved targets
        tk = self.kt.filter(pl.col("idx2").is_in(c["idx2"].unique().implode()))
        full = c.select("idx", "idx2").join(q.select("idx", "key", "w", "typ"), on="idx") \
                .join(tk, on=["idx2", "key"], how="semi")
        agg = full.group_by("idx", "idx2").agg(
            R(pl.col("w").filter(pl.col("typ") <= 3).sum()).alias("score"),
            pl.col("w").filter(pl.col("typ") <= 1).sum().alias("s_name"),
            pl.col("w").filter(pl.col("typ") == 2).sum().alias("s_addr"),
            (pl.col("typ") == 3).sum().cast(pl.Float32).alias("s_dig"),
            pl.col("w").filter(pl.col("typ").is_in([4, 5, 9])).sum().alias("s_phon"),
            pl.col("w").filter(pl.col("typ").is_in([6, 7, 8])).sum().alias("s_comp"),
            (pl.col("typ") == 6).sum().cast(pl.Float32).alias("hnpc_shared"),
            pl.col("w").filter(pl.col("typ") == 10).sum().alias("s_ngn"),
            pl.col("w").filter(pl.col("typ") == 11).sum().alias("s_nga"),
        )
        c = c.join(agg, on=["idx", "idx2"], how="left").fill_null(0).with_columns(
            R(pl.col("s_phon") + pl.col("s_comp")).alias("_sb"),
            R(pl.col("s_ngn") + pl.col("s_nga")).alias("_sc"))
        c = c.sort(["idx", "score", "s1score", "idx2"], descending=[False, True, True, False]) \
             .with_columns(pl.int_range(pl.len()).over("idx").alias("rank"))
        c = c.sort(["idx", "_sb", "s1b", "idx2"], descending=[False, True, True, False]) \
             .with_columns(pl.int_range(pl.len()).over("idx").alias("rank_b"))
        c = c.sort(["idx", "_sc", "s1c", "idx2"], descending=[False, True, True, False]) \
             .with_columns(pl.int_range(pl.len()).over("idx").alias("rank_c"))
        c = c.filter((pl.col("rank") < K)
                     | ((pl.col("rank_b") < K_b) & (pl.col("_sb") > 0))
                     | ((pl.col("rank_c") < K_c) & (pl.col("_sc") > 0)))
        return c.drop("_sb", "_sc").sort(["idx", "rank"])

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
            g(S, "tot_ng", i1, "tng1"), g(T, "tot_ng", i2, "tng2"),
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
            pl.col("s_phon").log1p().alias("s_phon_l"), pl.col("s_comp").log1p().alias("s_comp_l"),
            pl.col("rank_b").cast(pl.Float32), pl.col("s1b").log1p().alias("s1b_l"),
            pl.col("rank_c").cast(pl.Float32), pl.col("s1c").log1p().alias("s1c_l"),
            pl.col("s_ngn").log1p().alias("s_ngn_l"), pl.col("s_nga").log1p().alias("s_nga_l"),
            div(pl.col("s_ngn"), pl.col("tng1")).alias("ng_cov1"), div(pl.col("s_ngn"), pl.col("tng2")).alias("ng_cov2"),
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

    def iter_chunks(self, K, K_pre, chunk, cost_cap=20_000_000, K_b=20, K_pre_b=50, K_c=10, K_pre_c=30):
        for lo, hi in self.chunk_bounds(chunk, cost_cap):
            c = self.candidates(lo, hi, K, K_pre, K_b, K_pre_b, K_c, K_pre_c)
            if c is not None and c.height:
                yield self.features(c)


def X_of(f):
    return f.select(FEATURES).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
