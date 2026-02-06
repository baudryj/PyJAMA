"""
Script 70_aggregated : 55_resampled → 70_aggregated pour PyJAMA.
Agrège par fenêtre temporelle (ex. 1h, 1d), une ligne par (ts_bucket, device_id, domain)
avec colonnes par (métrique, méthode) : median, average. Métrique absente = ignorée (pas d'erreur).
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


def get_domain_from_path(path: Path) -> Optional[str]:
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.parquet -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_base_stem_from_resampled_stem(resampled_stem: str) -> str:
    """Dérive le base_stem en retirant le suffixe _resampled_* du nom du fichier resampled."""
    if "_resampled_" in resampled_stem:
        return resampled_stem.split("_resampled_")[0]
    return resampled_stem


def get_file_timestamp_range(path: Path, timestamp_column: str = "ts") -> Tuple[Optional[datetime], Optional[datetime]]:
    """Lit la plage temporelle (min/max) selon le format : Parquet ou CSV (colonne timestamp)."""
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
        first_ts_str = first_line.split(",")[0]
        last_ts_str = last_line.split(",")[0]
        return parse_ts(first_ts_str), parse_ts(last_ts_str)
    except Exception as e:
        logger.warning(f"Erreur lecture plage temporelle de {path}: {e}")
        return None, None


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


def _normalize_metrics_aggregation(metrics_aggregation: Any) -> List[Tuple[str, str, str]]:
    """Convertit metrics_aggregation (dict ou liste) en liste de (metric, method, output).

    - metric : nom de la metric (ex. m6)
    - method : 'median' ou 'average'
    - output : nom de la colonne en sortie (par défaut == metric pour éviter les renommages).
    """
    result: List[Tuple[str, str, str]] = []
    if isinstance(metrics_aggregation, dict):
        for metric, specs in metrics_aggregation.items():
            for spec in specs:
                method = spec.get("method", "average")
                # Par défaut on garde le même nom de colonne que la metric
                output = spec.get("output", metric)
                result.append((metric, method, output))
    elif isinstance(metrics_aggregation, list):
        for item in metrics_aggregation:
            metric = item.get("metric", "")
            method = item.get("method", "average")
            output = item.get("output", metric)
            result.append((metric, method, output))
    return result


def _build_metrics_aggregation_from_domain(
    df: pl.DataFrame,
    domain: str,
    aggregation_by_domain: Dict[str, Dict[str, Any]],
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Construit dynamiquement une liste (metric, method, output) à partir de aggregation_by_domain
    en gardant les noms de colonnes identiques aux metrics.

    aggregation_by_domain attendu :
    {
      "bio_signal": { "method": "median" },
      "environment": { "method": "average" }
    }
    """
    domain_cfg = aggregation_by_domain.get(domain)
    if not domain_cfg:
        return None

    method = domain_cfg.get("method", "average")

    if "metric" not in df.columns:
        return None

    present_metrics = sorted(set(df["metric"].unique().to_list()))
    specs: List[Tuple[str, str, str]] = []
    for metric in present_metrics:
        specs.append((metric, method, metric))
    return specs


def _aggregate_single_file(input_path: Path, config: Dict) -> Tuple[Optional[pl.DataFrame], int]:
    """
    Lit un Parquet/CSV long, agrège par ts_bucket + device_id + domain selon metrics_aggregation.
    Retourne (DataFrame agrégé wide, rows_before) ou (None, 0) en cas d'erreur.
    Métriques absentes : ignorées (pas d'erreur).
    """
    try:
        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=",", null_values="NaN")
    except Exception as e:
        logger.warning(f"Impossible de lire {input_path}: {e}")
        return None, 0

    rows_before = df.height
    ts_col_in = config.get("input", {}).get("timestamp_column", "ts")
    required = [ts_col_in, "device_id", "metric", "value", "domain"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"Colonne {col} absente dans {input_path.name}, fichier ignoré")
            return None, rows_before
    if ts_col_in != "ts":
        df = df.rename({ts_col_in: "ts"})

    aggregation_level = config.get("aggregation_level", "1h")
    metrics_aggregation = config.get("metrics_aggregation", {})
    aggregation_by_domain = config.get("aggregation_by_domain") or {}

    specs: List[Tuple[str, str, str]] = []
    if metrics_aggregation:
        # Mode explicite : on suit strictement metrics_aggregation
        specs = _normalize_metrics_aggregation(metrics_aggregation)
    elif aggregation_by_domain:
        # Mode simplifié par domaine : même méthode pour toutes les metrics du domaine,
        # en conservant les noms de colonnes identiques aux metrics.
        specs_domain = _build_metrics_aggregation_from_domain(df, get_domain_from_path(input_path) or "", aggregation_by_domain)
        if specs_domain:
            specs = specs_domain

    if not specs:
        logger.warning("Aucune règle d'agrégation trouvée (metrics_aggregation et aggregation_by_domain vides ou non applicables), fichier ignoré")
        return None, rows_before

    # ts -> datetime puis ts_bucket (début de fenêtre)
    if df.schema["ts"] == pl.Utf8:
        df = df.with_columns(
            pl.col("ts").str.to_datetime(time_zone="UTC", strict=False).alias("ts_dt")
        )
    else:
        df = df.with_columns(pl.col("ts").alias("ts_dt"))
    df = df.with_columns(pl.col("ts_dt").dt.truncate(aggregation_level).alias("ts_bucket"))

    present_metrics = set(df["metric"].unique().to_list())

    # #region agent log
    _debug_log = Path(__file__).resolve().parent.parent / "logs" / "debug.log"
    try:
        import json
        _debug_log.parent.mkdir(parents=True, exist_ok=True)
        requested = [s[0] for s in specs]
        with open(_debug_log, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"hypothesisId": "H1-H3", "location": "70_aggregated.py:_aggregate_single_file", "message": "present_vs_requested_metrics", "data": {"path": str(input_path), "present_metrics": sorted(present_metrics), "requested_metrics": requested, "match": [m for m in requested if m in present_metrics]}, "timestamp": int(datetime.utcnow().timestamp() * 1000), "sessionId": "debug-session"}) + "\n")
    except Exception:
        pass
    # #endregion

    # Clés uniques (ts_bucket, device_id, domain)
    keys = df.select(["ts_bucket", "device_id", "domain"]).unique()
    result = keys

    for metric, method, output_name in specs:
        if metric not in present_metrics:
            logger.debug(f"Métrique {metric} absente dans {input_path.name}, colonne {output_name} non créée")
            continue
        agg_expr = pl.col("value").median() if method == "median" else pl.col("value").mean()
        agg_df = (
            df.filter(pl.col("metric") == metric)
            .group_by(["ts_bucket", "device_id", "domain"])
            .agg(agg_expr.alias(output_name))
        )
        result = result.join(agg_df, on=["ts_bucket", "device_id", "domain"], how="left")

    result = result.rename({"ts_bucket": "ts"})
    # #region agent log
    try:
        import json
        with open(_debug_log, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"hypothesisId": "H1", "location": "70_aggregated.py:_aggregate_single_file", "message": "result_columns_after_agg", "data": {"path": str(input_path), "columns": list(result.columns)}, "timestamp": int(datetime.utcnow().timestamp() * 1000), "sessionId": "debug-session"}) + "\n")
    except Exception:
        pass
    # #endregion
    return result, rows_before


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Lit un fichier 55_resampled, agrège, écrit en Parquet dans 70_aggregated/domain=*/.
    Retourne (chemin_sortie, rapport).
    """
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
            report["error"] = "Impossible d'extraire le domaine du chemin"
            return None, report

        agg_df, rows_before = _aggregate_single_file(input_path, config)
        if agg_df is None:
            report["error"] = "Agrégation impossible (fichier ignoré ou erreur)"
            return None, report

        report["rows_before"] = rows_before
        report["rows_after"] = agg_df.height

        output_cfg = config["output"]
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        base_stem = get_base_stem_from_resampled_stem(input_path.stem)
        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_aggregated_{NOW_DATETIME}").strip()
        now_str = datetime.utcnow().strftime("%Y.%m.%d.T.%H.%M.%SZ")
        suffix = suffix_template.replace("{NOW_DATETIME}", now_str)
        p = (prefix + "_") if prefix else ""
        s = (suffix if suffix and suffix.startswith("_") else ("_" + suffix) if suffix else "")
        output_filename = f"{p}{base_stem}{s}.parquet"
        domain_dir = output_dir / f"domain={domain}"
        domain_dir.mkdir(parents=True, exist_ok=True)
        output_path = domain_dir / output_filename

        ts_col_out = output_cfg.get("timestamp_column") or "ts"
        if ts_col_out != "ts":
            agg_df = agg_df.rename({"ts": ts_col_out})
        agg_df = format_timestamp_column_utc_z(agg_df, ts_col_out)

        decimal_places = output_cfg.get("decimal_places")
        for col in agg_df.columns:
            if col in (ts_col_out, "device_id", "domain"):
                continue
            if agg_df.schema[col] in (pl.Float32, pl.Float64):
                n = _precision_for_column(decimal_places, col)
                if n is not None:
                    agg_df = agg_df.with_columns(pl.col(col).round(n).alias(col))

        output_columns = config.get("output", {}).get("output_columns") or config.get("output_columns")
        agg_df = apply_output_columns(agg_df, output_columns)

        compression = output_cfg.get("compression", "snappy")
        agg_df.write_parquet(output_path, compression=compression)
        report["output_file"] = str(output_path)
        logger.info(f"  Écrit {output_path} ({agg_df.height} lignes)")

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return output_path, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 55_resampled → 70_aggregated pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama (total_files, processed_files, failed_files, total_rows_*).
    """
    logger.info("Début 70_aggregated (55_resampled → 70_aggregated)")

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

    file_pattern = input_cfg.get("file_pattern", "*.parquet")
    search_subdirs = input_cfg.get("search_in_subdirectory", "no").lower() == "yes"
    except_pattern = input_cfg.get("except", "") or ""

    if search_subdirs:
        all_files = list(input_dir.rglob(file_pattern))
    else:
        all_files = list(input_dir.glob(file_pattern))
    all_files = [f for f in all_files if f.suffix.lower() in (".parquet", ".csv")]
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
            ts_col = input_cfg.get("timestamp_column", "ts")
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

    aggregation_by_domain = config.get("aggregation_by_domain") or {}

    for file_path in filtered_files:
        # Si on est en mode aggregation_by_domain, on peut avoir des fichiers
        # appartenant à des domaines non configurés (ex. environment) : on les
        # ignore silencieusement (info) sans les compter en échec.
        domain = get_domain_from_path(file_path)
        if aggregation_by_domain and (not domain or domain not in aggregation_by_domain):
            logger.info(
                f"Fichier {file_path} ignoré : domaine '{domain}' sans règle d'agrégation dans aggregation_by_domain"
            )
            # #region agent log
            try:
                import json
                from pathlib import Path as _Path
                _dbg_path = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                _dbg_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_dbg_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "pre-fix",
                                "hypothesisId": "H-agg-domain-skip",
                                "location": "70_aggregated.py:run",
                                "message": "skip_file_without_domain_rule",
                                "data": {
                                    "path": str(file_path),
                                    "domain": domain,
                                    "configured_domains": sorted(aggregation_by_domain.keys()),
                                },
                                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            continue

        output_path, report = process_single_file(file_path, config)
        if report.get("error"):
            summary["failed_files"].append(report)
        else:
            summary["processed_files"].append(report)
            summary["total_rows_before"] += report.get("rows_before", 0)
            summary["total_rows_after"] += report.get("rows_after", 0)

    logger.info(
        f"70_aggregated terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
