"""
Utilitaire Parquet → CSV pour PyJAMA.
Lit un ou plusieurs fichiers Parquet, applique optionnellement decimal_places (arrondi puis format string),
normalise les timestamps en UTC Z, et écrit en CSV.
À utiliser en dehors du pipeline de transformation (pipeline = full Parquet).
"""

from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import logging
import polars as pl

from format_ts import format_timestamp_column_utc_z

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

STAGED_CSV_SEP = ","


def _precision_for_column(decimal_places: Optional[Any], col_name: str) -> Optional[int]:
    """Précision décimale pour une colonne : int (toutes) ou dict {col: n, 'default': n}."""
    if decimal_places is None:
        return None
    if isinstance(decimal_places, int):
        return decimal_places
    if isinstance(decimal_places, dict):
        if col_name in decimal_places:
            return int(decimal_places[col_name])
        default = decimal_places.get("default")
        return int(default) if default is not None else None
    return None


def _precision_for_metric(decimal_places: Optional[Any], metric_name: str) -> Optional[int]:
    """Précision décimale pour une metric (format long)."""
    return _precision_for_column(decimal_places, metric_name)


def _convert_one(
    path: Path, input_dir: Path, output_dir: Path, decimal_places: Optional[Any], ts_col: str
) -> Optional[Path]:
    """Convertit un fichier Parquet en CSV. Préserve la structure de répertoires (ex. domain=*). Retourne le chemin du CSV écrit ou None."""
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        logger.warning(f"Impossible de lire {path}: {e}")
        return None

    if ts_col in df.columns:
        df = format_timestamp_column_utc_z(df, ts_col)

    # Schéma long (metric, value) ou large (une colonne par métrique)
    has_metric = "metric" in df.columns and "value" in df.columns
    if has_metric:
        for m in df["metric"].unique().to_list():
            n = _precision_for_metric(decimal_places, m)
            if n is not None:
                fmt_str = "{:." + str(n) + "f}"
                df = df.with_columns(
                    pl.when(pl.col("metric") == m)
                    .then(
                        pl.col("value").map_elements(
                            lambda x, f=fmt_str: f.format(x) if x is not None else "NaN",
                            return_dtype=pl.Utf8,
                        )
                    )
                    .otherwise(pl.col("value").cast(pl.Utf8))
                    .alias("value")
                )
        df = df.with_columns(pl.col("value").cast(pl.Utf8).fill_null("NaN"))
    else:
        for col in df.columns:
            if col == ts_col:
                continue
            if df.schema[col] in (pl.Float32, pl.Float64):
                n = _precision_for_column(decimal_places, col)
                if n is not None:
                    fmt_str = "{:." + str(n) + "f}"
                    df = df.with_columns(
                        pl.col(col).map_elements(
                            lambda x, f=fmt_str: f.format(x) if x is not None else "NaN",
                            return_dtype=pl.Utf8,
                        )
                    )

    try:
        rel = path.relative_to(input_dir)
    except ValueError:
        rel = path.name
    out_path = output_dir / Path(str(rel)).with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path, separator=STAGED_CSV_SEP, null_value="NaN")
    logger.info(f"  Écrit {out_path} ({df.height} lignes)")
    return out_path


def run(config: Dict) -> Dict:
    """
    Point d'entrée : convertit les Parquet d'un répertoire (ou liste) en CSV.
    Config attendue :
      input: input_directory, file_pattern ("*.parquet"), search_in_subdirectory ("yes"/"no")
      output: output_directory, decimal_places (optionnel, int ou { col/metric: n, "default": n })
      timestamp_column (optionnel): "Time" ou "ts" (défaut "Time" si colonne existe, sinon "ts")
    Retourne un résumé : total_files, converted_files, failed_files.
    """
    logger.info("Début parquet_to_csv")
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    input_dir = Path(input_cfg.get("input_directory", "."))
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    output_dir = Path(output_cfg.get("output_directory", "data/csv_export"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    file_pattern = input_cfg.get("file_pattern", "*.parquet")
    search_subdirs = input_cfg.get("search_in_subdirectory", "yes").lower() == "yes"
    decimal_places = output_cfg.get("decimal_places")
    ts_col = config.get("timestamp_column", "Time")

    if search_subdirs:
        all_files = list(input_dir.rglob(file_pattern))
    else:
        all_files = list(input_dir.glob(file_pattern))
    all_files = [f for f in all_files if f.suffix.lower() == ".parquet"]
    logger.info(f"{len(all_files)} fichier(s) Parquet trouvé(s)")

    converted = 0
    for path in all_files:
        out = _convert_one(path, input_dir, output_dir, decimal_places, ts_col)
        if out is not None:
            converted += 1

    summary = {
        "total_files": len(all_files),
        "converted_files": converted,
        "failed_files": len(all_files) - converted,
    }
    logger.info(f"Terminé : {converted}/{len(all_files)} fichier(s) converti(s)")
    return summary


if __name__ == "__main__":
    import json
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "input": {"input_directory": "data/PREMANIP_GRACE/55_resampled", "file_pattern": "*.parquet", "search_in_subdirectory": "yes"},
            "output": {"output_directory": "data/PREMANIP_GRACE/csv_export", "decimal_places": {"default": 3}},
            "timestamp_column": "ts",
        }
    result = run(cfg)
    print(result)
