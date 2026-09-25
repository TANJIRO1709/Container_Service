"""Country-agnostic text normalization (Phase 1: hand-written dictionaries;
Phase 2 replaces/extends them with rewrite rules mined from the ground truth).

Output columns per record:
  idx (u32 row index), entity_id, cty, name_key, addr_key, name_toks, addr_toks, name_ntok
"""
import polars as pl

# Legal / generic suffixes dropped from NAMES (kept if the name would become empty).
LEGAL = {
    # US / generic English
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp",
    "ltd", "limited", "plc", "pllc", "pc", "the", "and", "of",
    # India
    "pvt", "private", "opc",
    # France / EU (test set has France; no training data, so hand-listed)
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "cie", "et", "gmbh",
}

NAME_ABBR = {
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "srvcs": "services", "service": "services",
    "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "ent": "enterprises", "enterprise": "enterprises", "tech": "technologies",
    "technology": "technologies", "sys": "systems", "system": "systems",
    "mgmt": "management", "dev": "development", "grp": "group", "ind": "industries",
    "industry": "industries", "natl": "national", "solns": "solutions", "solution": "solutions",
}

ADDR_ABBR = {
    # English
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite",
    "fl": "floor", "flr": "floor", "apt": "apartment", "bldg": "building", "blk": "block",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "mkt": "market", "sec": "sector", "ngr": "nagar",
    "clny": "colony", "cly": "colony", "extn": "extension", "ext": "extension",
    "mg": "mahatma gandhi",
    # French
    "r": "rue", "che": "chemin", "rte": "route", "imp": "impasse",
    "fbg": "faubourg", "all": "allee",
}

DROP_NAME = sorted(LEGAL)

# canonical legal form (for legal_eq / legal_conflict features)
LEGAL_CANON = {
    "pvt": "pvt", "private": "pvt", "ltd": "ltd", "limited": "ltd", "inc": "inc",
    "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh",
}


def clean_expr(col: str) -> pl.Expr:
    return (
        pl.col(col).fill_null("")
        .str.normalize("NFKD").str.replace_all(r"\p{M}+", "")   # strip accents: é -> e
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"['’`]", "")                          # o'neil -> oneil
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_row_index("idx").with_columns(
        pl.col("country").fill_null("").str.strip_chars().str.to_lowercase().alias("cty"),
        clean_expr("business_name").str.replace(r"^m s ", "").alias("_n"),   # "M/s Sharma" -> "sharma"
        clean_expr("business_address").alias("_a"),
    )
    out = out.with_columns(
        pl.col("_n").str.split(" ")
        .list.eval(pl.element().replace(NAME_ABBR).filter(pl.element() != "")).alias("_nt_raw"),
        pl.col("_a").str.split(" ")
        .list.eval(pl.element().replace(ADDR_ABBR).filter(pl.element() != ""))
        .list.join(" ").str.split(" ")          # re-split multi-word expansions ("mahatma gandhi")
        .list.eval(pl.element().filter(pl.element() != ""))
        .alias("addr_toks"),
    )
    out = out.with_columns(
        pl.col("_nt_raw").list.eval(pl.element().filter(~pl.element().is_in(DROP_NAME))).alias("_nt")
    ).with_columns(
        pl.when(pl.col("_nt").list.len() > 0).then(pl.col("_nt")).otherwise(pl.col("_nt_raw")).alias("name_toks")
    )
    digits_only = pl.element().str.contains(r"^\d+$")
    out = out.with_columns(
        # postal code = last all-digit token of length 5-6 (US ZIP, India PIN, France CP)
        pl.col("addr_toks").list.eval(pl.element().filter(digits_only & pl.element().str.len_chars().is_between(5, 6)))
        .list.last().alias("pc"),
        pl.col("addr_toks").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).alias("_dig"),
        pl.col("_nt_raw").list.eval(pl.element().replace_strict(LEGAL_CANON, default=None).drop_nulls())
        .list.unique().list.sort().list.join(" ").alias("legal"),
    ).with_columns(
        # house number = first digit-bearing token that is not the postal code
        pl.col("_dig").list.eval(pl.element()).list.first().alias("_d0"),
    ).with_columns(
        pl.when(pl.col("_d0") == pl.col("pc")).then(None).otherwise(pl.col("_d0")).alias("hn"),
        pl.col("pc").str.slice(0, 3).alias("pc3"),
        pl.col("_dig").list.sort().list.join(" ").alias("dig_key"),
        pl.col("name_toks").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("initials"),
        pl.col("name_toks").list.first().alias("first_tok"),
    )
    return out.select(
        "idx", "entity_id", "cty",
        pl.col("name_toks").list.join(" ").alias("name_key"),
        pl.col("addr_toks").list.join(" ").alias("addr_key"),
        "name_toks", "addr_toks",
        pl.col("name_toks").list.len().cast(pl.Float32).alias("name_ntok"),
        pl.col("addr_toks").list.len().cast(pl.Float32).alias("addr_ntok"),
        pl.col("_n").alias("name_raw"),
        "pc", "pc3", "hn", "dig_key", "legal", "initials", "first_tok",
    )
