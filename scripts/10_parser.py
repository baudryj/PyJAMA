"""
Script 10_parser : raw → staged pour PyJAMA.
Parse les CSV iCaging (séparateur ;, colonnes key:value), normalise les timestamps,
applique une déduplication basique et écrit en CSV ou Parquet dans 10_staged.
"""

from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import re
import shutil
from fnmatch import fnmatch

import polars as pl

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from output_columns_helper import apply_output_columns

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _precision_for_column(decimal_places: Optional[Any], col_name: str) -> Optional[int]:
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


def parse_line_to_dict(line: str) -> Dict[str, Any]:
    """
    Transforme une ligne CSV iCaging en dictionnaire.
    Format attendu: timestamp;m10:7;m11:7;...;outdoor_temp:18.81
    """
    parts = line.strip().split(";")
    if not parts:
        return {}

    result = {"Time": parts[0]}
    for part in parts[1:]:
        if ":" in part:
            col_name, value = part.split(":", 1)
            try:
                result[col_name] = float(value) if "." in value else int(value)
            except ValueError:
                result[col_name] = value
    return result


def load_csv_custom(path: Path) -> pl.DataFrame:
    """Charge un fichier CSV iCaging (séparateur ;, colonnes key:value)."""
    logger.info(f"Chargement du fichier: {path}")
    data_rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row_dict = parse_line_to_dict(line)
                if row_dict:
                    data_rows.append(row_dict)
            except Exception as e:
                logger.warning(f"Erreur ligne {line_num} dans {path}: {e}")
                continue

    if not data_rows:
        raise ValueError(f"Aucune donnée valide trouvée dans {path}")

    df = pl.DataFrame(data_rows)
    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def normalize_timestamps(df: pl.DataFrame, time_column: str = "Time") -> pl.DataFrame:
    """Normalise les timestamps au format ISO (sans modifier le temps)."""
    if time_column not in df.columns:
        raise ValueError(f"Colonne {time_column} non trouvée")

    logger.info("Normalisation des timestamps")
    time_formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
    ]

    def normalize_ts(ts_str: str) -> str:
        if ts_str is None:
            return ts_str
        for fmt in time_formats:
            try:
                dt = datetime.strptime(str(ts_str), fmt)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                continue
        try:
            dt = datetime.fromisoformat(str(ts_str).replace("Z", ""))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return str(ts_str)

    df = df.with_columns(
        pl.col(time_column).map_elements(normalize_ts, return_dtype=pl.Utf8)
    )
    invalid_count = df.select(pl.col(time_column).is_null().sum()).to_series(0)[0]
    if invalid_count > 0:
        logger.warning(f"{invalid_count} timestamps invalides trouvés")
    logger.info(f"Timestamps normalisés: {df.height} lignes")
    return df


def apply_rules_staged(df: pl.DataFrame, rules: Dict) -> pl.DataFrame:
    """
    Applique sélection de colonnes, types et min/max → null.
    Pas de forward_fill ni drop_nulls (staged reste proche du raw).
    """
    rules = rules or {}
    if not rules:
        logger.info("rules_and_filters vide : toutes les colonnes conservées.")
        return df

    columns_to_keep = ["Time"] + list(rules.keys())
    existing = [c for c in columns_to_keep if c in df.columns]
    missing = [c for c in columns_to_keep if c not in df.columns]
    if missing:
        logger.warning(f"Colonnes absentes des données: {missing}")

    df = df.select(existing)
    logger.info(f"Colonnes conservées: {existing}")

    for col_name, rule in rules.items():
        if col_name not in df.columns:
            continue
        if "type" in rule:
            target_type = rule["type"]
            if target_type == "int":
                df = df.with_columns(pl.col(col_name).cast(pl.Int64, strict=False))
            elif target_type == "float":
                df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
        if "min" in rule:
            min_val = rule["min"]
            df = df.with_columns(
                pl.when(pl.col(col_name) < min_val)
                .then(None)
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
        if "max" in rule:
            max_val = rule["max"]
            df = df.with_columns(
                pl.when(pl.col(col_name) > max_val)
                .then(None)
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
    return df


def dedup(df: pl.DataFrame, subset: Optional[List[str]] = None) -> pl.DataFrame:
    """Supprime les lignes dupliquées (toutes les colonnes ou subset)."""
    n_before = df.height
    if subset is not None and subset:
        df = df.unique(subset=subset)
    else:
        df = df.unique()
    n_after = df.height
    if n_before > n_after:
        logger.info(f"Déduplication: {n_before} → {n_after} lignes ({n_before - n_after} doublons supprimés)")
    return df


def get_file_timestamp_range(path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Lit première et dernière ligne pour obtenir la plage temporelle du fichier."""
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

        first_ts_str = first_line.split(";")[0]
        last_ts_str = last_line.split(";")[0]
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"]
        first_ts = last_ts = None
        for fmt in formats:
            try:
                first_ts = datetime.strptime(first_ts_str, fmt)
                break
            except ValueError:
                continue
        for fmt in formats:
            try:
                last_ts = datetime.strptime(last_ts_str, fmt)
                break
            except ValueError:
                continue
        return first_ts, last_ts
    except Exception as e:
        logger.warning(f"Erreur lecture plage temporelle de {path}: {e}")
        return None, None


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Traite un fichier raw : load → normalize_ts → dedup → rules (optionnel) → write CSV.
    Retourne (chemin_sortie, rapport) avec rapport au format attendu par pyjama.
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
        df = load_csv_custom(input_path)
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column", "Time")
        if ts_col_in not in df.columns and "Time" in df.columns:
            ts_col_in = "Time"
        df = normalize_timestamps(df, time_column=ts_col_in)

        input_cfg = config.get("input", {})
        if input_cfg.get("filter_by_file_date", True):
            m = re.match(r"^.*_(\d{4}-\d{2}-\d{2})$", input_path.stem)
            if m:
                expected_date = m.group(1)
                n_before = df.height
                df = df.filter(pl.col(ts_col_in).str.slice(0, 10) == expected_date)
                removed = n_before - df.height
                if removed > 0:
                    logger.info(f"{removed} lignes supprimées ({ts_col_in} != date du fichier {expected_date})")
                if df.height == 0:
                    logger.warning(f"Aucune ligne restante après filtrage par date {expected_date}, pas d'écriture")
                    report["rows_after"] = 0
                    report["output_file"] = None
                    return None, report

        if input_cfg.get("dedup", True):
            dedup_cols = config.get("output", {}).get("dedup_columns")
            df = dedup(df, subset=dedup_cols)

        rules = config.get("rules_and_filters", {})
        if rules:
            df = apply_rules_staged(df, rules)

        output_cfg = config["output"]
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix = (output_cfg.get("output_file_suffix") or "").strip()
        now_str = datetime.utcnow().strftime("%Y.%m.%d.T.%H.%M.%SZ")
        suffix_with_now = suffix.replace("{NOW_DATETIME}", now_str)

        def build_filename() -> str:
            stem = input_path.stem
            sub_cfg = output_cfg.get("file_name_substitute")
            if isinstance(sub_cfg, dict):
                src = sub_cfg.get("src") or ""
                target = sub_cfg.get("target") or ""
                if src:
                    stem = stem.replace(src, target)
            p = (prefix + "_") if prefix else ""
            s = ("_" + suffix_with_now) if suffix_with_now else ""
            return f"{p}{stem}{s}.parquet"

        partition_by = output_cfg.get("partition_by") or []
        compression = output_cfg.get("compression")
        ts_col_out = output_cfg.get("timestamp_column") or ts_col_in
        if ts_col_out != ts_col_in:
            df = df.rename({ts_col_in: ts_col_out})

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

        if partition_by:
            df = df.with_columns(
                pl.col(ts_col_out).str.slice(0, 10).alias("day")
            )
            base_filename = build_filename()
            days = df["day"].unique().to_list()
            for day_val in days:
                day_str = day_val if isinstance(day_val, str) else str(day_val)
                df_day = df.filter(pl.col("day") == day_val).drop("day")
                part_dir = output_dir / f"day={day_str}"
                part_dir.mkdir(parents=True, exist_ok=True)
                out_path = part_dir / base_filename
                df_day.write_parquet(out_path, compression=compression)
                logger.info(f"  Écrit {out_path} ({df_day.height} lignes)")
            output_path = output_dir
            report["output_file"] = str(output_dir)
            logger.info(f"Fichier traité avec succès: {output_dir} (partitionné par {partition_by}, {df.height} lignes)")
        else:
            output_filename = build_filename()
            output_path = output_dir / output_filename
            df.write_parquet(output_path, compression=compression)
            report["output_file"] = str(output_path)
            logger.info(f"Fichier traité avec succès: {output_path} ({df.height} lignes)")

        report["rows_after"] = df.height

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return output_path, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : parse raw → staged pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama (total_files, processed_files, failed_files,
    total_rows_before, total_rows_after).
    """
    logger.info("Début 10_parser (raw → staged)")

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
            time_filtered = []
            for f in filtered_files:
                first_ts, last_ts = get_file_timestamp_range(f)
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
        f"10_parser terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
