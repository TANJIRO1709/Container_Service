"""Final corpus: Phase-6/7 lexical blocking (channels A, B, C) + dense channel D
+ cross-country fallback channel E.

Channel E: every other channel only compares records carrying the SAME country label.
If a true pair's labels disagree ("US" vs "USA", missing, noisy), no channel can ever
retrieve it. Channel E indexes a few very RARE country-free keys — adjacent name-token
pairs, the full normalized name, house-number|postal-code — and keeps only CROSS-country
hits (same-country ones are already covered), top K_e per S1. Rare-only keeps it cheap
and precise; the model sees `cty_eq` and learns how much evidence a cross-country pair needs.


Channel D candidates are ADDED to the lexical union (never displacing it). Every final
candidate, whichever channel found it, gets the full stage-2 IDF overlap and the full
feature set, plus three dense features:
  dcos    cosine similarity of the two record embeddings (computed for EVERY candidate)
  rank_d  rank of the target in this S1's dense neighbour list (999 = not retrieved by D)
  d_only  1 if only channel D retrieved it
"""
import numpy as np
import polars as pl

from dense import embed, knn, serialize
from pipeline6 import FEATURES as FEATURES6
from pipeline6 import Corpus6

DENSE_FEATURES = ["dcos", "rank_d", "d_only"]
GLOBAL_FEATURES = ["gscore_l", "x_only", "cty_eq"]
NOT_RETRIEVED = 999


class Corpus8(Corpus6):
    def __init__(self, s1_raw, tgt_raw, dense_model=None, K_d=20, emb_cache=None, tag="x",
                 dense_batch=512, log=print, global_x=False, K_e=5, max_df_e=20, **kw):
        super().__init__(s1_raw, tgt_raw, log=log, **kw)
        self.dense = bool(dense_model)
        self.K_d = K_d
        self.gx = bool(global_x)
        if self.gx:
            self._build_global(K_e, max_df_e, log)
        if not self.dense:
            return
        log(f"  dense channel: model={dense_model}  K_d={K_d}")
        q = embed(serialize(self.s1["name_key"].fill_null("").to_list(), self.s1["addr_key"].fill_null("").to_list()),
                  dense_model, emb_cache, f"{tag}_s1", batch=dense_batch, log=log)
        t = embed(serialize(self.tg["name_key"].fill_null("").to_list(), self.tg["addr_key"].fill_null("").to_list()),
                  dense_model, emb_cache, f"{tag}_tg", batch=dense_batch, log=log)
        codes = {c: i for i, c in enumerate(sorted(set(self.s1["cty"].to_list()) | set(self.tg["cty"].to_list())))}
        qc = np.array([codes[c] for c in self.s1["cty"].to_list()], dtype=np.int32)
        tc = np.array([codes[c] for c in self.tg["cty"].to_list()], dtype=np.int32)
        qi, tj, sim, rk = knn(q, t, qc, tc, K_d, log=log)
        self.dn = pl.DataFrame({"idx": qi.astype(np.uint32), "idx2": tj.astype(np.uint32),
                                "rank_d": rk.astype(np.int32)}).sort("idx")
        self._dn_idx = self.dn["idx"]
        self.q_emb, self.t_emb = q, t
        log(f"  dense neighbours: {self.dn.height:,}")

    # ------------------------------------------------------------------ channel E
    @staticmethod
    def _gkeys(df, batch=1_000_000):
        out = []
        for o in range(0, df.height, batch):
            b = df.slice(o, batch)
            t = b.select("idx", pl.col("name_key").fill_null("").str.split(" ").alias("tok")).explode("tok") \
                 .filter(pl.col("tok").str.len_chars() >= 2)
            t = t.with_columns(pl.col("tok").shift(-1).over("idx").alias("nxt"))
            bi = t.filter(pl.col("nxt").is_not_null()).select("idx", ("b|" + pl.col("tok") + "_" + pl.col("nxt")).alias("k"))
            nm = b.filter(pl.col("name_key").fill_null("").str.len_chars() >= 6) \
                  .select("idx", ("n|" + pl.col("name_key")).alias("k"))
            hp = b.filter(pl.col("hn").is_not_null() & pl.col("pc").is_not_null()) \
                  .select("idx", ("h|" + pl.col("hn") + "|" + pl.col("pc")).alias("k"))
            out.append(pl.concat([bi, nm, hp]).select("idx", pl.col("k").hash(seed=77).alias("key")).unique())
        return pl.concat(out)

    def _build_global(self, K_e, max_df_e, log, step=200_000):
        kt = self._gkeys(self.tg).rename({"idx": "idx2"})
        dft = kt.group_by("key").agg(pl.len().alias("dft")).filter(pl.col("dft") <= max_df_e)
        nt = self.tg.height
        dft = dft.with_columns((((nt + 1) / (pl.col("dft") + 1)).log()).cast(pl.Float32).alias("w"))
        kt = kt.join(dft.select("key"), on="key", how="semi")
        k1 = self._gkeys(self.s1).join(dft.select("key", "w"), on="key")
        s1c, tgc = self.s1["cty"], self.tg["cty"]
        res = []
        for lo in range(0, self.s1.height, step):
            q = k1.filter((pl.col("idx") >= lo) & (pl.col("idx") < lo + step))
            p = q.join(kt, on="key").group_by("idx", "idx2").agg(pl.col("w").sum().round(4).alias("gscore"))
            if p.height == 0:
                continue
            p = p.filter(s1c.gather(p["idx"]) != tgc.gather(p["idx2"]))           # cross-country only
            p = p.sort(["idx", "gscore", "idx2"], descending=[False, True, False]) \
                 .filter(pl.int_range(pl.len()).over("idx") < K_e)
            res.append(p)
        self.gxp = (pl.concat(res) if res else
                    pl.DataFrame(schema={"idx": pl.UInt32, "idx2": pl.UInt32, "gscore": pl.Float32})).sort("idx")
        self._gx_idx = self.gxp["idx"]
        log(f"  cross-country fallback (channel E): {self.gxp.height:,} candidate pairs")

    # ------------------------------------------------------------------ helpers
    def _overlap(self, pairs, lo, hi):
        """Stage-2 IDF overlap for arbitrary (idx, idx2) pairs — same maths as Corpus6."""
        a = int(self._k1_idx.search_sorted(lo, side="left"))
        b = int(self._k1_idx.search_sorted(hi, side="left"))
        q = self.k1.slice(a, b - a)
        R = lambda e: e.round(4)
        tk = self.kt.filter(pl.col("idx2").is_in(pairs["idx2"].unique().implode()))
        full = pairs.select("idx", "idx2").join(q.select("idx", "key", "w", "typ"), on="idx") \
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
        return pairs.join(agg, on=["idx", "idx2"], how="left").fill_null(0)

    def _cos(self, idx, idx2, step=2_000_000):
        out = np.empty(len(idx), dtype=np.float32)
        for s in range(0, len(idx), step):
            a = self.q_emb[idx[s:s + step]].astype(np.float32)
            b = self.t_emb[idx2[s:s + step]].astype(np.float32)
            out[s:s + step] = (a * b).sum(1)
        return out

    # ------------------------------------------------------------------ blocking
    def candidates(self, lo, hi, K, K_pre, K_b=20, K_pre_b=50, K_c=10, K_pre_c=30):
        c = super().candidates(lo, hi, K, K_pre, K_b, K_pre_b, K_c, K_pre_c)
        if not (self.dense or self.gx):
            return c

        def sl(df, col):
            x, y = int(col.search_sorted(lo, side="left")), int(col.search_sorted(hi, side="left"))
            return df.slice(x, y - x)

        d = sl(self.dn, self._dn_idx) if self.dense else None
        e = sl(self.gxp, self._gx_idx) if self.gx else None
        ext = [z.select("idx", "idx2") for z in (d, e) if z is not None and z.height]
        if c is not None and c.height == 0:
            c = None
        if ext:
            new = pl.concat(ext).unique()
            if c is not None:
                new = new.join(c.select("idx", "idx2"), on=["idx", "idx2"], how="anti")
            if new.height:
                new = self._overlap(new, lo, hi).with_columns(
                    pl.lit(0.0).alias("s1score"), pl.lit(0.0).alias("s1b"), pl.lit(0.0).alias("s1c"),
                    pl.lit(NOT_RETRIEVED).alias("rank"), pl.lit(NOT_RETRIEVED).alias("rank_b"),
                    pl.lit(NOT_RETRIEVED).alias("rank_c"), pl.lit(1).alias("_ext"))
                c = new if c is None else pl.concat([c.with_columns(pl.lit(0).alias("_ext")), new],
                                                    how="diagonal_relaxed")
        if c is None or c.height == 0:
            return None
        if "_ext" not in c.columns:
            c = c.with_columns(pl.lit(0).alias("_ext"))
        c = c.with_columns(pl.col("_ext").fill_null(0))
        if self.dense:
            c = c.join(d.select("idx", "idx2", "rank_d"), on=["idx", "idx2"], how="left") \
                 .with_columns(pl.col("rank_d").fill_null(NOT_RETRIEVED))
            c = c.with_columns(pl.Series("dcos", self._cos(c["idx"].to_numpy(), c["idx2"].to_numpy())),
                               ((pl.col("_ext") == 1) & (pl.col("rank_d") < NOT_RETRIEVED)).cast(pl.Float32)
                               .alias("d_only"))
        if self.gx:
            c = c.join(e.select("idx", "idx2", "gscore"), on=["idx", "idx2"], how="left") \
                 .with_columns(pl.col("gscore").fill_null(0.0))
            in_d = (pl.col("rank_d") < NOT_RETRIEVED) if self.dense else pl.lit(False)
            c = c.with_columns(
                ((pl.col("_ext") == 1) & (pl.col("gscore") > 0) & ~in_d).cast(pl.Float32).alias("x_only"),
                (self.s1["cty"].gather(c["idx"]) == self.tg["cty"].gather(c["idx2"])).cast(pl.Float32).alias("cty_eq"))
        return c.drop("_ext").sort(["idx", "rank"])

    def feature_list(self):
        return feature_list(self.dense, self.gx)

    def features(self, c):
        f = super().features(c)
        cols = (["dcos", "rank_d", "d_only"] if self.dense else []) + (["gscore", "x_only", "cty_eq"] if self.gx else [])
        if not cols:
            return f
        f = f.join(c.select("idx", "idx2", *[pl.col(x).cast(pl.Float32) for x in cols]), on=["idx", "idx2"], how="left")
        return f.with_columns(pl.col("gscore").log1p().alias("gscore_l")) if self.gx else f


def feature_list(dense, gx=False):
    return FEATURES6 + (DENSE_FEATURES if dense else []) + (GLOBAL_FEATURES if gx else [])
