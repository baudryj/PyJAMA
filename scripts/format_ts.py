"""
Helper : normalise la colonne timestamp (Time ou ts) au format UTC avec suffixe Z.
Format de sortie : YYYY-MM-DDTHH:MM:SSZ (string).
Assure la cohérence du format des timestamps à travers toutes les étapes du pipeline.
"""

from datetime import datetime
from typing import Optional

import polars as pl


TS_FORMAT_Z = "%Y-%m-%dT%H:%M:%SZ"
TS_FORMATS_PARSE = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
]


def _normalize_ts_str(s: Optional[str]) -> Optional[str]:
    """Parse un timestamp string et retourne le format YYYY-MM-DDTHH:MM:SSZ."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    s = str(s).strip()
    for fmt in TS_FORMATS_PARSE:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime(TS_FORMAT_Z)
        except (ValueError, TypeError):
            continue
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime(TS_FORMAT_Z)
    except Exception:
        return s


def format_timestamp_column_utc_z(df: pl.DataFrame, col_name: str = "ts") -> pl.DataFrame:
    """
    Normalise la colonne timestamp au format string UTC avec Z (YYYY-MM-DDTHH:MM:SSZ).
    Gère colonne string ou datetime. Les null restent null.
    """
    if col_name not in df.columns:
        return df
    dtype = df.schema[col_name]
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col(col_name).map_elements(_normalize_ts_str, return_dtype=pl.Utf8).alias(col_name)
        )
    if isinstance(dtype, pl.Datetime):
        return df.with_columns(
            pl.col(col_name).dt.strftime(TS_FORMAT_Z).alias(col_name)
        )
    return df
