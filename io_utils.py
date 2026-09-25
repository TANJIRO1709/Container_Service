"""Shared IO helpers. All columns are read as strings; quoting disabled
(addresses may contain stray quote characters)."""
import polars as pl

READ_KW = dict(separator="\t", quote_char=None, infer_schema_length=0)


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, **READ_KW)


def scan_tsv(path: str) -> pl.LazyFrame:
    return pl.scan_csv(path, **READ_KW)


def explode_ids(df: pl.DataFrame, list_col: str, out_col: str = "target_id") -> pl.DataFrame:
    """(source1_entity_id, 'a,b,c') -> one row per (source1_entity_id, id).
    Empty/null lists are dropped (singletons disappear here, by design)."""
    return (
        df.select(
            pl.col("source1_entity_id"),
            pl.col(list_col).fill_null("").str.split(",").alias(out_col),
        )
        .explode(out_col)
        .with_columns(pl.col(out_col).str.strip_chars())
        .filter(pl.col(out_col).is_not_null() & (pl.col(out_col) != ""))
    )


def write_id_lists(path: str, header2: str, s1_ids, id_lists: dict) -> None:
    """Write submission-format TSV. id_lists: {s1_id: [ids]} (missing -> empty).
    Deduplicates while preserving order. Written manually so empty cells are truly empty."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for sid in s1_ids:
            ids = id_lists.get(sid, ())
            seen, out = set(), []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            f.write(f"{sid}\t{','.join(out)}\n")
