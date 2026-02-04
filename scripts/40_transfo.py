"""
Script 40_transfo : 30_clean → 40_transfo pour PyJAMA.
Applique des transformations par domaine (sqrt_inv, log, celsius_to_fahrenheit)
et ajoute de nouvelles colonnes. La config référence des types de transformation
implémentés dans le script.
"""

from pathlib import Path
import sys
from datetime import datetime

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from typing import Dict, Optional, Tuple
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


def _precision_for_column(decimal_places, col_name: str):
    """Retourne la précision décimale pour une colonne. decimal_places peut être int (toutes colonnes) ou dict {col: n, 'default': n}."""
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


def get_domain_from_path(path: Path) -> Optional[str]:
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.csv -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_base_stem_from_clean_stem(clean_stem: str) -> str:
    """Dérive le base_stem en retirant le suffixe _clean_* du nom du fichier clean."""
    if "_clean_" in clean_stem:
        return clean_stem.split("_clean_")[0]
    return clean_stem


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


def _apply_one_transfo(col_expr: pl.Expr, transfo_type: str) -> pl.Expr:
    """
    Retourne l'expression Polars pour une transformation donnée.
    col_expr = pl.col(nom_colonne_source).
    """
    if transfo_type == "sqrt_inv":
        # sqrt(1/x) pour x > 0, sinon NaN
        return pl.when(col_expr > 0).then((1.0 / col_expr) ** 0.5).otherwise(None)
    if transfo_type == "log":
        # ln(x) pour x > 0, sinon NaN
        return pl.when(col_expr > 0).then(col_expr.log()).otherwise(None)
    if transfo_type == "celsius_to_fahrenheit":
        # F = C * 9/5 + 32 (NaN propagé)
        return col_expr * (9.0 / 5.0) + 32.0
    raise ValueError(f"Type de transformation inconnu: {transfo_type}")


def apply_domain_transfo(df: pl.DataFrame, domain_transfo: Dict) -> pl.DataFrame:
    """
    Pour chaque colonne source listée dans domain_transfo, applique les transformations
    (out_col -> type) et ajoute les nouvelles colonnes. Les colonnes existantes sont conservées.
    """
    if not domain_transfo:
        return df

    for source_col, out_columns in domain_transfo.items():
        if source_col not in df.columns:
            logger.warning(f"Colonne source absente: {source_col}")
            continue
        col_expr = pl.col(source_col)
        for out_col, transfo_type in out_columns.items():
            try:
                expr = _apply_one_transfo(col_expr, transfo_type)
                df = df.with_columns(expr.alias(out_col))
                logger.debug(f"  {source_col} -> {out_col} ({transfo_type})")
            except ValueError as e:
                logger.warning(f"Transfo ignorée {source_col} -> {out_col}: {e}")
    return df


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Lit un CSV ou Parquet depuis 30_clean/domain=*/, applique les transformations du domaine,
    écrit dans 40_transfo/domain=*/ (CSV ou Parquet selon config).
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

        domain_transfo = config.get("domain_transfo", {})
        transfo = domain_transfo.get(domain, {})

        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=STAGED_CSV_SEP, null_values="NaN")
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column", "Time")
        if ts_col_in not in df.columns:
            raise ValueError(f"Colonne timestamp absente dans {input_path.name}: {ts_col_in}")

        df = apply_domain_transfo(df, transfo)
        report["rows_after"] = df.height

        output_cfg = config["output"]
        ts_col_out = output_cfg.get("timestamp_column") or ts_col_in
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        base_stem = get_base_stem_from_clean_stem(input_path.stem)
        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_transfo_{NOW_DATETIME}").strip()
        now_str = datetime.utcnow().strftime("%Y.%m.%d.T.%H.%M.%SZ")
        suffix = suffix_template.replace("{NOW_DATETIME}", now_str)
        p = (prefix + "_") if prefix else ""
        s = (suffix if suffix and suffix.startswith("_") else ("_" + suffix) if suffix else "")
        output_filename = f"{p}{base_stem}{s}.parquet"
        domain_dir = output_dir / f"domain={domain}"
        domain_dir.mkdir(parents=True, exist_ok=True)
        output_path = domain_dir / output_filename

        if ts_col_out != ts_col_in:
            df = df.rename({ts_col_in: ts_col_out})
        df = format_timestamp_column_utc_z(df, ts_col_out)
        decimal_places = output_cfg.get("decimal_places")
        for col in df.columns:
            if col == ts_col_out:
                continue
            if df.schema[col] in (pl.Float32, pl.Float64):
                n = _precision_for_column(decimal_places, col)
                if n is not None:
                    df = df.with_columns(pl.col(col).round(n).alias(col))
        output_columns = output_cfg.get("output_columns")
        df = apply_output_columns(df, output_columns)
        compression = output_cfg.get("compression", "snappy")
        df.write_parquet(output_path, compression=compression)
        report["output_file"] = str(output_path)
        logger.info(f"  Écrit {output_path} ({df.height} lignes)")

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return output_path, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : transfo 30_clean → 40_transfo pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama.
    """
    logger.info("Début 40_transfo (30_clean → 40_transfo)")

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
            ts_col = input_cfg.get("timestamp_column", "Time")
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
        f"40_transfo terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
