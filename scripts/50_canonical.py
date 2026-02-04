"""
Script 50_canonical : 40_transfo → 50_canonical pour PyJAMA.
Passe du format large (une colonne par métrique) au schéma long canonique :
ts, device_id, metric, value, unit, domain, quality_flag.
"""

from pathlib import Path
import sys
from datetime import datetime

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from typing import Any, Dict, List, Optional, Tuple
import logging
import shutil
from fnmatch import fnmatch

import polars as pl

from format_ts import format_timestamp_column_utc_z
from output_columns_helper import apply_output_columns

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

STAGED_CSV_SEP = ","


def _precision_for_metric(decimal_places: Optional[Any], metric_name: str) -> Optional[int]:
    """Retourne la précision décimale pour une metric. decimal_places peut être int (toutes metrics) ou dict {metric: n, 'default': n}."""
    if decimal_places is None:
        return None
    if isinstance(decimal_places, int):
        return decimal_places
    if isinstance(decimal_places, dict):
        if metric_name in decimal_places:
            return int(decimal_places[metric_name])
        default = decimal_places.get("default")
        return int(default) if default is not None else None
    return None


def get_domain_from_path(path: Path) -> Optional[str]:
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.csv -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_base_stem_from_transfo_stem(transfo_stem: str) -> str:
    """Dérive le base_stem en retirant le suffixe _transfo_* du nom du fichier transfo."""
    if "_transfo_" in transfo_stem:
        return transfo_stem.split("_transfo_")[0]
    return transfo_stem


def get_file_timestamp_range(path: Path, timestamp_column: str = "Time") -> Tuple[Optional[datetime], Optional[datetime]]:
    """Lit la plage temporelle (min/max) selon le format : Parquet ou CSV."""
    formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"]

    def parse_ts(s: Optional[str]) -> Optional[datetime]:
        if s is None:
            return None
        for fmt in formats:
            try:
                return datetime.strptime(str(s).strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None

    if path.suffix.lower() == ".parquet":
        try:
            row = pl.scan_parquet(path).select(
                pl.col(timestamp_column).min().alias("min_ts"),
                pl.col(timestamp_column).max().alias("max_ts"),
            ).collect()
            if row.height == 0:
                return None, None
            min_ts_str = row["min_ts"][0]
            max_ts_str = row["max_ts"][0]
            return parse_ts(min_ts_str), parse_ts(max_ts_str)
        except Exception as e:
            logger.warning(f"Erreur lecture plage temporelle Parquet de {path}: {e}")
            return None, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None, None
            last_line = first_line
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        first_ts_str = first_line.split(STAGED_CSV_SEP)[0]
        last_ts_str = last_line.split(STAGED_CSV_SEP)[0]
        return parse_ts(first_ts_str), parse_ts(last_ts_str)
    except Exception as e:
        logger.warning(f"Erreur lecture plage temporelle de {path}: {e}")
        return None, None


def wide_to_canonical(
    df: pl.DataFrame,
    value_columns: List[str],
    timestamp_column: str,
    quality_flag_column: str,
    device_id: str,
    domain: str,
    unit_map: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Passe du format large au long canonique : ts, device_id, metric, value, unit, domain, quality_flag.
    value_columns : colonnes à fondre (métriques). Seules celles présentes dans df sont utilisées.
    """
    unit_map = unit_map or {}
    present = [c for c in value_columns if c in df.columns]
    if not present:
        raise ValueError(f"Aucune colonne de valeur trouvée parmi {value_columns}")

    id_vars = [timestamp_column]
    if quality_flag_column in df.columns:
        id_vars.append(quality_flag_column)

    molten = df.select(id_vars + present).melt(
        id_vars=id_vars,
        value_vars=present,
        variable_name="metric",
        value_name="value",
    )

    molten = molten.rename({timestamp_column: "ts"})
    if quality_flag_column not in df.columns:
        molten = molten.with_columns(pl.lit(None).cast(pl.Int64).alias(quality_flag_column))

    molten = molten.with_columns(
        pl.lit(device_id).alias("device_id"),
        pl.lit(domain).alias("domain"),
        pl.when(pl.col("metric").is_in(list(unit_map.keys())))
        .then(pl.col("metric").replace(unit_map))
        .otherwise(pl.lit(""))
        .alias("unit"),
    )

    canonical_order = ["ts", "device_id", "metric", "value", "unit", "domain", quality_flag_column]
    return molten.select([c for c in canonical_order if c in molten.columns])


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Lit un CSV ou Parquet depuis 40_transfo/domain=*/, convertit en long canonique,
    écrit dans 50_canonical/domain=*/ (CSV ou Parquet selon config).
    """
    logger.info(f"Traitement du fichier: {input_path.name}")
    report = {
        "input_file": str(input_path),
        "output_file": None,
        "rows_before": 0,
        "rows_after": 0,
        "error": None,
    }

    try:
        domain = get_domain_from_path(input_path)
        if not domain:
            raise ValueError(f"Impossible d'extraire le domaine du chemin: {input_path}")

        domain_columns = config.get("domain_columns", {})
        value_columns = domain_columns.get(domain, [])
        if not value_columns:
            raise ValueError(f"Aucune colonne configurée pour le domaine: {domain}")

        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=STAGED_CSV_SEP, null_values="NaN")
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column") or config.get("timestamp_column", "Time")
        if ts_col_in not in df.columns:
            raise ValueError(f"Colonne timestamp absente: {ts_col_in}")

        device_id = config.get("device_id", "")
        unit_map = config.get("unit_map", {})

        long_df = wide_to_canonical(
            df,
            value_columns=value_columns,
            timestamp_column=ts_col_in,
            quality_flag_column="quality_flag",
            device_id=device_id,
            domain=domain,
            unit_map=unit_map,
        )
        report["rows_after"] = long_df.height

        output_cfg = config["output"]
        ts_col_out = output_cfg.get("timestamp_column") or "ts"
        if ts_col_out != "ts":
            long_df = long_df.rename({"ts": ts_col_out})

        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        base_stem = get_base_stem_from_transfo_stem(input_path.stem)
        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_canonical_{NOW_DATETIME}").strip()
        now_str = datetime.utcnow().strftime("%Y.%m.%d.T.%H.%M.%SZ")
        suffix = suffix_template.replace("{NOW_DATETIME}", now_str)
        p = (prefix + "_") if prefix else ""
        s = (suffix if suffix and suffix.startswith("_") else ("_" + suffix) if suffix else "")
        output_filename = f"{p}{base_stem}{s}.parquet"
        domain_dir = output_dir / f"domain={domain}"
        domain_dir.mkdir(parents=True, exist_ok=True)
        output_path = domain_dir / output_filename

        long_df = format_timestamp_column_utc_z(long_df, ts_col_out)
        decimal_places = output_cfg.get("decimal_places")
        for m in long_df["metric"].unique().to_list():
            n = _precision_for_metric(decimal_places, m)
            if n is not None:
                long_df = long_df.with_columns(
                    pl.when(pl.col("metric") == m)
                    .then(pl.col("value").round(n))
                    .otherwise(pl.col("value"))
                    .alias("value")
                )
        output_columns = output_cfg.get("output_columns")
        long_df = apply_output_columns(long_df, output_columns)
        compression = output_cfg.get("compression", "snappy")
        long_df.write_parquet(output_path, compression=compression)
        report["output_file"] = str(output_path)
        logger.info(f"  Écrit {output_path} ({long_df.height} lignes)")

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return output_path, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 40_transfo → 50_canonical pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama.
    """
    logger.info("Début 50_canonical (40_transfo → 50_canonical)")

    output_cfg = config.get("output", {})
    if output_cfg.get("clear"):
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info("Répertoire de sortie vidé (clear=true)")

    input_cfg = config.get("input", {})
    input_dir = Path(input_cfg["input_directory"])
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    file_pattern = input_cfg.get("file_pattern", "*.csv")
    search_subdirs = input_cfg.get("search_in_subdirectory", "no").lower() == "yes"
    except_pattern = input_cfg.get("except", "") or ""

    if search_subdirs:
        all_files = list(input_dir.rglob(file_pattern))
    else:
        all_files = list(input_dir.glob(file_pattern))

    logger.info(f"{len(all_files)} fichiers trouvés avec pattern {file_pattern}")

    if except_pattern:
        filtered_files = [f for f in all_files if not fnmatch(f.name, except_pattern)]
        logger.info(f"{len(filtered_files)} fichiers après exclusion de '{except_pattern}'")
    else:
        filtered_files = all_files

    from_date = input_cfg.get("from", "")
    to_date = input_cfg.get("to", "")
    if from_date or to_date:

        def parse_date(date_str: str, is_end: bool = False) -> Optional[datetime]:
            if not date_str or date_str == "yyyy-mm-ddTh:m:sZ":
                return None
            if len(date_str) == 10 and date_str.count("-") == 2:
                date_str = f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"
            for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None

        from_dt = parse_date(from_date, is_end=False)
        to_dt = parse_date(to_date, is_end=True)

        if from_dt or to_dt:
            ts_col = input_cfg.get("timestamp_column") or config.get("timestamp_column", "Time")
            time_filtered = []
            for f in filtered_files:
                first_ts, last_ts = get_file_timestamp_range(f, timestamp_column=ts_col)
                if first_ts is None or last_ts is None:
                    logger.warning(f"Plage temporelle illisible pour {f}, fichier inclus")
                    time_filtered.append(f)
                    continue
                include = True
                if from_dt and last_ts < from_dt:
                    include = False
                if to_dt and first_ts > to_dt:
                    include = False
                if include:
                    time_filtered.append(f)
                else:
                    logger.info(f"Fichier {f.name} exclu (hors plage temporelle)")
            filtered_files = time_filtered
            logger.info(f"{len(filtered_files)} fichiers après filtrage temporel")

    summary = {
        "total_files": len(filtered_files),
        "processed_files": [],
        "failed_files": [],
        "total_rows_before": 0,
        "total_rows_after": 0,
    }

    for file_path in filtered_files:
        output_path, report = process_single_file(file_path, config)
        if report["error"]:
            summary["failed_files"].append(report)
        else:
            summary["processed_files"].append(report)
            summary["total_rows_before"] += report["rows_before"]
            summary["total_rows_after"] += report["rows_after"]

    logger.info(
        f"50_canonical terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
