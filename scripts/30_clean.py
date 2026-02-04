"""
Script 30_clean : 20_split → 30_clean pour PyJAMA.
Applique les règles de nettoyage par domaine (type, min/max, max_diff spike).
Spike : valeur remplacée par NaN. Ajoute quality_flag (int) : 0=OK, 1=Manquant, 2=Spike, 3=Hors plage, 4=Capteur déconnecté.
"""

from pathlib import Path
import sys
from datetime import datetime

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from typing import Dict, List, Tuple, Optional
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


def get_domain_from_path(path: Path) -> Optional[str]:
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.csv -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_base_stem_from_split_stem(split_stem: str) -> str:
    """Dérive le base_stem en retirant le suffixe _split_<domain>_* du nom du fichier split."""
    if "_split_" in split_stem:
        return split_stem.split("_split_")[0]
    return split_stem


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


def apply_domain_rules(
    df: pl.DataFrame,
    rules: Dict,
    add_quality_column: bool,
) -> pl.DataFrame:
    """
    Applique les règles par colonne : type, min/max, max_diff (spike -> valeur précédente).
    Ajoute la colonne quality si add_quality_column (priorité : missing > impossible > outlier > spike > ok).
    """
    if not rules:
        if add_quality_column:
            return df.with_columns(pl.lit("ok").alias("quality"))
        return df

    # Manques initiaux : au moins une colonne nulle dans la ligne
    has_missing = pl.any_horizontal(pl.all().is_null())

    # Colonnes temporaires pour cumuler les masques par ligne
    df = df.with_columns(
        pl.lit(False).alias("_hi"),
        pl.lit(False).alias("_ho"),
        pl.lit(False).alias("_hs"),
    )

    for col_name, rule in rules.items():
        if col_name not in df.columns:
            continue

        # 1) Cast de type (sauver l'état avant pour impossible)
        before_cast = df.select(pl.col(col_name)).to_series()
        if rule.get("type") == "int":
            df = df.with_columns(pl.col(col_name).cast(pl.Int64, strict=False).alias(col_name))
        elif rule.get("type") == "float":
            df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name))
        after_cast = df.select(pl.col(col_name)).to_series()
        impossible_mask = before_cast.is_not_null() & after_cast.is_null()
        df = df.with_columns(
            (pl.col("_hi") | impossible_mask).alias("_hi")
        )

        # 2) Min / max (hors plage -> null)
        before_minmax = df.select(pl.col(col_name)).to_series()
        if "min" in rule:
            min_val = rule["min"]
            df = df.with_columns(
                pl.when(pl.col(col_name) < min_val).then(None).otherwise(pl.col(col_name)).alias(col_name)
            )
        if "max" in rule:
            max_val = rule["max"]
            df = df.with_columns(
                pl.when(pl.col(col_name) > max_val).then(None).otherwise(pl.col(col_name)).alias(col_name)
            )
        after_minmax = df.select(pl.col(col_name)).to_series()
        outlier_mask = before_minmax.is_not_null() & after_minmax.is_null()
        df = df.with_columns(
            (pl.col("_ho") | outlier_mask).alias("_ho")
        )

        # 3) Spike : |valeur(i) - valeur(i-1)| > max_diff -> remplacer par NaN
        if "max_diff" in rule:
            max_diff = rule["max_diff"]
            prev = pl.col(col_name).shift(1)
            diff = (pl.col(col_name) - prev).abs()
            spike_mask = (diff > max_diff) & diff.is_finite() & prev.is_not_null()
            df = df.with_columns(
                pl.when(spike_mask).then(None).otherwise(pl.col(col_name)).alias(col_name)
            )
            df = df.with_columns(
                (pl.col("_hs") | spike_mask).alias("_hs")
            )

    if add_quality_column:
        # quality_flag : int (0=OK, 1=Manquant, 2=Spike, 3=Hors plage, 4=Capteur déconnecté)
        quality_flag = (
            pl.when(has_missing)
            .then(pl.lit(1))   # Manquant
            .when(pl.col("_hi"))
            .then(pl.lit(4))   # Capteur déconnecté (impossible / type invalide)
            .when(pl.col("_ho"))
            .then(pl.lit(3))   # Hors plage physique
            .when(pl.col("_hs"))
            .then(pl.lit(2))   # Spike / saut brutal
            .otherwise(pl.lit(0))  # OK
        )
        df = df.with_columns(quality_flag.cast(pl.Int64).alias("quality_flag"))

    df = df.drop(["_hi", "_ho", "_hs"])
    return df


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Lit un CSV ou Parquet depuis 20_split/domain=*/, applique les règles du domaine,
    écrit dans 30_clean/domain=*/ (CSV ou Parquet selon config).
    Retourne (chemin sortie, rapport) au format attendu par pyjama.
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

        domain_rules = config.get("domain_rules", {})
        rules = domain_rules.get(domain, {})

        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=STAGED_CSV_SEP, null_values="NaN")
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column", "Time")
        if ts_col_in not in df.columns:
            raise ValueError(f"Colonne timestamp absente dans {input_path.name}: {ts_col_in}")

        output_cfg = config["output"]
        ts_col_out = output_cfg.get("timestamp_column") or ts_col_in
        add_quality = output_cfg.get("add_quality_column", True)
        df = apply_domain_rules(df, rules, add_quality)
        report["rows_after"] = df.height

        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        base_stem = get_base_stem_from_split_stem(input_path.stem)
        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_clean_{NOW_DATETIME}").strip()
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
    Point d'entrée : clean 20_split → 30_clean pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama.
    """
    logger.info("Début 30_clean (20_split → 30_clean)")

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
        f"30_clean terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
