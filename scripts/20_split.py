"""
Script 20_split : staged → split par domaine pour PyJAMA.
Lit les CSV ou Parquet staged, route vers des sous-dossiers domain=<nom>/ avec
uniquement les colonnes mappées par domaine (sans transformation).
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

# Séparateur des CSV staged (sortie 10_parser)
STAGED_CSV_SEP = ","


def get_base_stem(staged_stem: str) -> str:
    """
    Dérive le base_stem en retirant le suffixe _staged_* du nom du fichier staged.
    Ex: premanip_grace_pil-85_2026-01-29_staged_2026.01.29.T.12.02.47Z -> premanip_grace_pil-85_2026-01-29
    """
    if "_staged_" in staged_stem:
        return staged_stem.split("_staged_")[0]
    return staged_stem


def get_base_stem_from_staged_path(path: Path) -> str:
    """
    Dérive un base_stem stable pour les noms de sortie split (même règle que staged).
    - Depuis le nom du fichier : retire le suffixe _staged_* ou *_YYYY.MM.DD.T.H.M.S.Z.
    - Ex: premanip_grace_pil-85_2026-01-21_staged_2026.01.30.T.14.06.03Z -> premanip_grace_pil-85_2026-01-21
    """
    stem = path.stem
    if "_staged_" in stem:
        return stem.split("_staged_")[0]
    if stem.startswith("staged_data"):
        stem = stem[len("staged_data") :].lstrip("_")
    parts = stem.split("_")
    if len(parts) >= 2 and "." in parts[-1] and "T" in parts[-1]:
        stem = "_".join(parts[:-1])
    return stem if stem else get_base_stem(path.stem)


def get_file_timestamp_range(path: Path, timestamp_column: str = "Time") -> Tuple[Optional[datetime], Optional[datetime]]:
    """Lit la plage temporelle (min/max) selon le format : Parquet ou CSV staged."""
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


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[List[Path]], Dict]:
    """
    Lit un CSV staged, route par domaine (Time + colonnes du domaine), écrit dans
    output_directory/domain={domain}/{base_stem}{suffix}.csv.
    Retourne (liste des chemins écrits, rapport) au format attendu par pyjama.
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
        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=STAGED_CSV_SEP)
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column", "Time")
        if ts_col_in not in df.columns:
            raise ValueError(f"Colonne timestamp absente dans {input_path.name}: {ts_col_in}")

        domain_columns = config.get("domain_columns", {})
        if not domain_columns:
            raise ValueError("domain_columns vide dans la config")

        output_cfg = config["output"]
        ts_col_out = output_cfg.get("timestamp_column") or ts_col_in
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_split_{domain}_{NOW_DATETIME}").strip()
        now_str = datetime.utcnow().strftime("%Y.%m.%d.T.%H.%M.%SZ")
        base_stem = get_base_stem_from_staged_path(input_path)

        total_rows_written = 0
        written_paths = []

        for domain, columns in domain_columns.items():
            cols_to_keep = [ts_col_in] + [c for c in columns if c in df.columns]
            missing = [c for c in columns if c not in df.columns]
            if missing:
                logger.warning(f"Domaine {domain}: colonnes absentes ignorées: {missing}")

            if len(cols_to_keep) <= 1:
                logger.warning(f"Domaine {domain}: aucune colonne de donnée présente, fichier ignoré")
                continue

            df_domain = df.select(cols_to_keep)
            domain_dir = output_dir / f"domain={domain}"
            domain_dir.mkdir(parents=True, exist_ok=True)
            suffix = suffix_template.replace("{domain}", domain).replace("{NOW_DATETIME}", now_str)
            p = (prefix + "_") if prefix else ""
            s = ("_" + suffix) if suffix else ""
            output_filename = f"{p}{base_stem}{s}.parquet"
            output_path = domain_dir / output_filename
            if ts_col_out != ts_col_in:
                df_domain = df_domain.rename({ts_col_in: ts_col_out})
            df_domain = format_timestamp_column_utc_z(df_domain, ts_col_out)
            output_columns = output_cfg.get("output_columns")
            df_domain = apply_output_columns(df_domain, output_columns)
            compression = output_cfg.get("compression", "snappy")
            df_domain.write_parquet(output_path, compression=compression)
            total_rows_written += df_domain.height
            written_paths.append(output_path)
            out_cols = [ts_col_out] + [c for c in columns if c in df.columns]
            logger.info(f"  Écrit {output_path} ({df_domain.height} lignes, colonnes: {out_cols})")

        report["rows_after"] = total_rows_written
        report["output_file"] = str(written_paths[0]) if written_paths else None
        if len(written_paths) > 1:
            report["output_files"] = [str(p) for p in written_paths]
        logger.info(f"Fichier traité avec succès: {len(written_paths)} domaine(s) écrit(s)")

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return written_paths, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : split staged → 20_split par domaine pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama (total_files, processed_files, failed_files,
    total_rows_before, total_rows_after).
    """
    logger.info("Début 20_split (staged → split par domaine)")

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
        written_paths, report = process_single_file(file_path, config)
        if report["error"]:
            summary["failed_files"].append(report)
        else:
            summary["processed_files"].append(report)
            summary["total_rows_before"] += report["rows_before"]
            summary["total_rows_after"] += report["rows_after"]

    logger.info(
        f"20_split terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
