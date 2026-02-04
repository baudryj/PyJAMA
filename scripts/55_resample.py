"""
Script 55_resample : 50_canonical → 55_resampled pour PyJAMA.
Grille temporelle régulière par domaine, timestamps manquants, interpolation optionnelle.
Ajoute is_interpolated, gap_duration ; quality_flag 5 = estimée (interpolée), 1 = manquant.
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
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.parquet -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_base_stem_from_canonical_stem(canonical_stem: str) -> str:
    """Dérive le base_stem en retirant le suffixe _canonical_* du nom du fichier canonical."""
    if "_canonical_" in canonical_stem:
        return canonical_stem.split("_canonical_")[0]
    return canonical_stem


def get_file_timestamp_range(path: Path, timestamp_column: str = "ts") -> Tuple[Optional[datetime], Optional[datetime]]:
    """Lit la plage temporelle (min/max) selon le format : Parquet ou CSV (colonne timestamp pour canonical)."""
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


def _ensure_ts_datetime(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """Cast ts to datetime if string (ISO avec Z -> UTC)."""
    if df.schema[ts_col] == pl.String:
        return df.with_columns(
            pl.col(ts_col).str.to_datetime(time_zone="UTC", strict=False).alias(ts_col)
        )
    return df


def _resample_one_metric(
    grid: pl.DataFrame,
    data_metric: pl.DataFrame,
    metric_name: str,
    unit_val: str,
    device_id: str,
    domain: str,
    interpolation: Optional[str],
    independent_by_day: bool = False,
) -> pl.DataFrame:
    """
    Aligne une métrique sur la grille (join_asof backward), ajoute is_interpolated, gap_duration, quality_flag.
    """
    if data_metric.height == 0:
        return pl.DataFrame()

    data_metric = data_metric.with_columns(pl.col("ts").alias("obs_ts"))
    joined = grid.join_asof(
        data_metric.select(["ts", "value", "quality_flag", "obs_ts"]),
        on="ts",
        strategy="backward",
    )

    # Optionnel : en mode independent_by_day, remplir le début de journée avec la première
    # valeur observée de la journée (sans jamais utiliser un jour précédent).
    if independent_by_day:
        first_obs = joined.filter(pl.col("obs_ts").is_not_null()).select(["value", "obs_ts"]).head(1)
        if first_obs.height > 0:
            first_value = first_obs["value"][0]
            first_obs_ts = first_obs["obs_ts"][0]
            # Pour tous les ts strictement avant la première observation, propager cette valeur.
            joined = joined.with_columns(
                pl.when(pl.col("ts") < first_obs_ts)
                .then(pl.lit(first_obs_ts))
                .otherwise(pl.col("obs_ts"))
                .alias("obs_ts"),
                pl.when(pl.col("ts") < first_obs_ts)
                .then(pl.lit(first_value))
                .otherwise(pl.col("value"))
                .alias("value"),
            )

    # is_interpolated = pas d'observation exacte à ce ts (ts != obs_ts ou obs_ts null)
    is_interp = (pl.col("ts") != pl.col("obs_ts")) | pl.col("obs_ts").is_null()
    # gap_duration en secondes (0 si observation exacte, sinon ts - obs_ts)
    gap_duration_expr = (pl.col("ts") - pl.col("obs_ts")).dt.total_seconds()
    gap_duration = pl.when(pl.col("obs_ts").is_null()).then(None).when(pl.col("ts") != pl.col("obs_ts")).then(
        gap_duration_expr.cast(pl.Float64)
    ).otherwise(0.0)
    # quality_flag : 1 = manquant (pas de backward match), 5 = interpolé, sinon conservé
    quality_flag = (
        pl.when(pl.col("obs_ts").is_null())
        .then(pl.lit(1))
        .when(is_interp)
        .then(pl.lit(5))
        .otherwise(pl.col("quality_flag"))
    )

    out = joined.with_columns(
        is_interp.alias("is_interpolated"),
        gap_duration.alias("gap_duration"),
        quality_flag.alias("quality_flag"),
        pl.lit(metric_name).alias("metric"),
        pl.lit(unit_val).alias("unit"),
        pl.lit(device_id).alias("device_id"),
        pl.lit(domain).alias("domain"),
    ).select(["ts", "device_id", "metric", "value", "unit", "domain", "quality_flag", "is_interpolated", "gap_duration"])

    if interpolation and interpolation != "forward":
        if interpolation == "linear":
            out = out.with_columns(
                pl.col("value").interpolate().alias("value")
            )
        elif interpolation == "nearest":
            out = out.with_columns(
                pl.col("value").fill_null(pl.col("value").forward_fill()).alias("value")
            )
            out = out.with_columns(
                pl.col("value").fill_null(pl.col("value").backward_fill()).alias("value")
            )

    return out


def resample_domain(
    df: pl.DataFrame,
    domain: str,
    freq: str,
    interpolation: Optional[str],
    device_id: str,
    independent_by_day: bool = False,
) -> pl.DataFrame:
    """
    Resample un DataFrame long canonique (ts, device_id, metric, value, unit, domain, quality_flag)
    sur une grille régulière de fréquence freq. Retourne le même schéma + is_interpolated, gap_duration.
    """
    df = _ensure_ts_datetime(df)
    if df.height == 0:
        return df

    # Mode classique : grille unique de min_ts à max_ts (comportement historique).
    if not independent_by_day:
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        if min_ts is None or max_ts is None:
            return df

        grid_series = pl.datetime_range(min_ts, max_ts, interval=freq, eager=True)
        grid = pl.DataFrame({"ts": grid_series})

        metrics = df["metric"].unique().to_list()
        if "unit" in df.columns:
            units = {r["metric"]: r["unit"] for r in df.select(["metric", "unit"]).unique().iter_rows(named=True)}
        else:
            units = {m: "" for m in metrics}

        parts: List[pl.DataFrame] = []
        for metric_name in metrics:
            data_metric = df.filter(pl.col("metric") == metric_name).sort("ts").select(["ts", "value", "quality_flag"])
            unit_val = units.get(metric_name, "")
            part = _resample_one_metric(
                grid,
                data_metric,
                metric_name,
                unit_val,
                device_id,
                domain,
                interpolation,
            )
            if part.height > 0:
                parts.append(part)
    else:
        # Mode indépendant par jour : pour chaque date, grille de début de journée à la
        # dernière observation de la journée, avec complétion du début via _resample_one_metric.
        df = df.sort("ts")
        df = df.with_columns(pl.col("ts").dt.date().alias("day"))
        metrics_all = df["metric"].unique().to_list()
        if "unit" in df.columns:
            units_all = {
                r["metric"]: r["unit"] for r in df.select(["metric", "unit"]).unique().iter_rows(named=True)
            }
        else:
            units_all = {m: "" for m in metrics_all}

        parts: List[pl.DataFrame] = []
        for day_val in df["day"].unique().to_list():
            df_day = df.filter(pl.col("day") == day_val)
            if df_day.height == 0:
                continue
            min_ts = df_day["ts"].min()
            max_ts = df_day["ts"].max()
            if min_ts is None or max_ts is None:
                continue
            grid_series = pl.datetime_range(min_ts, max_ts, interval=freq, eager=True)
            grid = pl.DataFrame({"ts": grid_series})

            metrics = df_day["metric"].unique().to_list()
            for metric_name in metrics:
                data_metric = (
                    df_day.filter(pl.col("metric") == metric_name)
                    .sort("ts")
                    .select(["ts", "value", "quality_flag"])
                )
                unit_val = units_all.get(metric_name, "")
                part = _resample_one_metric(
                    grid,
                    data_metric,
                    metric_name,
                    unit_val,
                    device_id,
                    domain,
                    interpolation,
                    independent_by_day=True,
                )
                if part.height > 0:
                    parts.append(part)

    if not parts:
        return pl.DataFrame(schema={
            "ts": pl.Datetime("us"),
            "device_id": pl.Utf8,
            "metric": pl.Utf8,
            "value": pl.Float64,
            "unit": pl.Utf8,
            "domain": pl.Utf8,
            "quality_flag": pl.Int64,
            "is_interpolated": pl.Boolean,
            "gap_duration": pl.Float64,
        })
    return pl.concat(parts)


def process_single_file(input_path: Path, config: Dict) -> Tuple[Optional[Path], Dict]:
    """
    Lit un CSV ou Parquet depuis 50_canonical/domain=*/, resample sur grille régulière par domaine,
    écrit dans 55_resampled/domain=*/ (CSV ou Parquet selon config).
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

        resample_by_domain = config.get("resample_by_domain", {})
        domain_cfg = resample_by_domain.get(domain)
        if not domain_cfg:
            logger.warning(f"Domaine {domain} absent de resample_by_domain, fichier ignoré")
            report["error"] = f"Domaine {domain} non configuré dans resample_by_domain"
            return None, report

        freq = domain_cfg.get("freq", "1m")
        interpolation = domain_cfg.get("interpolation")
        independent_by_day = domain_cfg.get("independent_by_day", False)

        if input_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(input_path)
        else:
            df = pl.read_csv(input_path, separator=STAGED_CSV_SEP, null_values="NaN")
        report["rows_before"] = df.height

        ts_col_in = config.get("input", {}).get("timestamp_column", "ts")
        if ts_col_in not in df.columns:
            raise ValueError(f"Colonne timestamp absente dans {input_path.name}: {ts_col_in}")
        if ts_col_in != "ts":
            df = df.rename({ts_col_in: "ts"})
        if "metric" not in df.columns or "value" not in df.columns:
            raise ValueError(f"Colonnes metric/value absentes dans {input_path.name}")

        device_id = df["device_id"][0] if "device_id" in df.columns and df.height > 0 else config.get("device_id", "")

        resampled = resample_domain(
            df,
            domain=domain,
            freq=freq,
            interpolation=interpolation,
            device_id=device_id,
            independent_by_day=independent_by_day,
        )
        report["rows_after"] = resampled.height

        output_cfg = config["output"]
        output_dir = Path(output_cfg["output_directory"])
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        base_stem = get_base_stem_from_canonical_stem(input_path.stem)
        prefix = (output_cfg.get("output_file_prefix") or "").strip()
        suffix_template = (output_cfg.get("output_file_suffix") or "_resampled_{NOW_DATETIME}").strip()
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
            resampled = resampled.rename({"ts": ts_col_out})
        resampled = format_timestamp_column_utc_z(resampled, ts_col_out)
        decimal_places = output_cfg.get("decimal_places")
        for m in resampled["metric"].unique().to_list():
            n = _precision_for_metric(decimal_places, m)
            if n is not None:
                resampled = resampled.with_columns(
                    pl.when(pl.col("metric") == m)
                    .then(pl.col("value").round(n))
                    .otherwise(pl.col("value"))
                    .alias("value")
                )
        output_columns = output_cfg.get("output_columns")
        resampled = apply_output_columns(resampled, output_columns)
        compression = output_cfg.get("compression", "snappy")
        resampled.write_parquet(output_path, compression=compression)
        report["output_file"] = str(output_path)
        logger.info(f"  Écrit {output_path} ({resampled.height} lignes)")

    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report["error"] = str(e)
        return None, report

    return output_path, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 50_canonical → 55_resampled pour tous les fichiers sélectionnés.
    Retourne un résumé compatible avec pyjama.
    """
    logger.info("Début 55_resample (50_canonical → 55_resampled)")

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

    for file_path in filtered_files:
        output_path, report = process_single_file(file_path, config)
        if report["error"]:
            summary["failed_files"].append(report)
        else:
            summary["processed_files"].append(report)
            summary["total_rows_before"] += report["rows_before"]
            summary["total_rows_after"] += report["rows_after"]

    logger.info(
        f"55_resample terminé: {len(summary['processed_files'])} traités, {len(summary['failed_files'])} échecs"
    )
    return summary
