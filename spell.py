"""Corpus-driven typo correction (SymSpell-style deletion neighbourhoods).

Insight: a correct word ("sharma") appears many times across S1+S2+S3, while each
random typo ("shamra", "sharam", "shrma") appears only once or twice. So for every
RARE token (freq <= f_lo) we look for a FREQUENT token (freq >= f_hi) in the same
country within edit distance ~1, and map rare -> frequent for BLOCKING keys only
(features still use the original text).

Edit-distance-1 test without comparing all pairs: two words are within one
substitution / insertion / deletion / adjacent transposition iff their sets
{word} U {word with one char deleted} intersect. We hash those variants and join —
cost is linear in vocabulary size, not quadratic.

Unsupervised and computed on the corpus being matched (train or test), using no
labels and no external data.
"""
import polars as pl


def _token_freq(frames, batch=1_000_000, min_len=4):
    parts = []
    for f in frames:
        for off in range(0, f.height, batch):
            b = f.slice(off, batch)
            t = pl.concat([
                b.select("cty", pl.col("name_toks").alias("tok")).explode("tok"),
                b.select("cty", pl.col("addr_toks").alias("tok")).explode("tok"),
            ]).drop_nulls("tok").filter(
                (pl.col("tok").str.len_chars() >= min_len) & ~pl.col("tok").str.contains(r"\d"))
            parts.append(t.group_by("cty", "tok").len("freq"))
    return pl.concat(parts).group_by("cty", "tok").agg(pl.col("freq").sum())


def _variants(df):
    dele = (df.with_columns(pl.int_ranges(0, pl.col("tok").str.len_chars()).alias("i")).explode("i")
            .with_columns((pl.col("tok").str.slice(0, pl.col("i")) + pl.col("tok").str.slice(pl.col("i") + 1)).alias("v"))
            .drop("i"))
    same = df.with_columns(pl.col("tok").alias("v"))
    return pl.concat([dele, same], how="diagonal_relaxed").with_columns(
        pl.concat_str(["cty", "v"], separator="|").hash(seed=7).alias("vh")).drop("v")


def build_corrections(frames, f_lo: int = 2, f_hi: int = 5, min_len: int = 4) -> pl.DataFrame:
    """Returns mapping DataFrame (cty, tok, fix)."""
    freq = _token_freq(frames, min_len=min_len)
    rare = freq.filter(pl.col("freq") <= f_lo).select("cty", "tok")
    good = freq.filter(pl.col("freq") >= f_hi).select("cty", pl.col("tok").alias("fix"), pl.col("freq").alias("ffreq"))
    if rare.height == 0 or good.height == 0:
        return pl.DataFrame(schema={"cty": pl.Utf8, "tok": pl.Utf8, "fix": pl.Utf8})
    rv = _variants(rare).select("cty", "tok", "vh")
    gv = _variants(good.rename({"fix": "tok"})).select(pl.col("tok").alias("fix"), "ffreq", "vh")
    m = rv.join(gv, on="vh").select("cty", "tok", "fix", "ffreq").unique()
    # pick the most frequent neighbour; deterministic tie-break on the string
    m = m.sort(["cty", "tok", "ffreq", "fix"], descending=[False, False, True, False]) \
         .group_by("cty", "tok", maintain_order=True).first()
    return m.select("cty", "tok", "fix")
