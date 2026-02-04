"""
Helper partagé : application de output_columns (sélection et renommage ancien->nouveau)
avant écriture Parquet. Utilisé par 10_parser, 20_split, 30_clean, 40_transfo, 50_canonical, 55_resample, 70_aggregated.
"""

from typing import List, Optional

import polars as pl


def apply_output_columns(
    df: pl.DataFrame, output_columns: Optional[List[str]]
) -> pl.DataFrame:
    """
    Applique output_columns : sélection des colonnes et optionnellement renommage (ancien->nouveau).
    Si output_columns est vide ou None, retourne df inchangé.
    Chaque élément est soit un nom de colonne (écrite telle quelle), soit "ancien->nouveau" (renommage).
    Les colonnes listées mais absentes du DataFrame sont ignorées (pas d'erreur).
    """
    if not output_columns:
        return df
    parsed: List[tuple] = []
    for item in output_columns:
        s = (item or "").strip()
        if not s:
            continue
        if "->" in s:
            parts = s.split("->", 1)
            source = parts[0].strip()
            output_name = parts[1].strip()
        else:
            source = s
            output_name = s
        if source in df.columns:
            parsed.append((source, output_name))
    if not parsed:
        return df
    return df.select([pl.col(src).alias(dst) for src, dst in parsed])
