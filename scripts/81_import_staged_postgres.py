"""
Script 81_import_staged_postgres : 10_staged (Parquet) → table PostgreSQL (long) pour PyJAMA.

Lit les fichiers staged (Parquet uniquement) dans 10_staged, normalise la colonne temps,
infère device_id depuis le nom de fichier, convertit en format long (Time, device_id, sensor, value),
applique une agrégation 10s/60s, puis insère en base Postgres pour visualisation Grafana.
Même principe que 81_import_raw_postgres mais sans CSV ni format raw "timestamp;capteur:valeur".
"""

from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import logging
from fnmatch import fnmatch

import polars as pl

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import importlib.util
_raw_spec = importlib.util.spec_from_file_location(
    "_raw81", _scripts_dir / "81_import_raw_postgres.py"
)
_raw81 = importlib.util.module_from_spec(_raw_spec)
_raw_spec.loader.exec_module(_raw81)

from format_ts import format_timestamp_column_utc_z
from device_id_helper import get_device_id_from_stem

logger = logging.getLogger(__name__)


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 10_staged (Parquet) → table PostgreSQL (long).
    Même logique que 81_import_raw_postgres mais uniquement Parquet, pas de parsing CSV/semicolon.
    """
    logger.info("Début 81_import_staged_postgres (10_staged → PostgreSQL)")

    input_cfg = config.get("input", {})
    input_dir = Path(input_cfg["input_directory"])
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    file_pattern = input_cfg.get("file_pattern", "*.parquet")
    search_subdirs = input_cfg.get("search_in_subdirectory", "yes").lower() == "yes"
    except_pattern = input_cfg.get("except", "") or ""
    ts_col = input_cfg.get("timestamp_column", "Time")

    if search_subdirs:
        all_files = list(input_dir.rglob(file_pattern))
    else:
        all_files = list(input_dir.glob(file_pattern))
    all_files = [f for f in all_files if f.suffix.lower() == ".parquet"]
    logger.info(f"{len(all_files)} fichiers Parquet trouvés avec pattern {file_pattern}")

    if except_pattern:
        filtered_files = [f for f in all_files if not fnmatch(f.name, except_pattern)]
        logger.info(f"{len(filtered_files)} fichiers après exclusion de '{except_pattern}'")
    else:
        filtered_files = all_files

    from_date = input_cfg.get("from", "")
    to_date = input_cfg.get("to", "")
    if from_date or to_date:
        filtered_files = _raw81._apply_time_filters_to_files(
            filtered_files, from_date, to_date, ts_col
        )

    if not filtered_files:
        logger.warning("Aucun fichier à traiter")
        return {
            "total_files": 0,
            "processed_files": [],
            "failed_files": [],
            "total_rows_before": 0,
            "total_rows_after": 0,
        }

    db_cfg = config.get("output", {}).get("database", {})
    if not db_cfg:
        logger.error("output.database manquant dans la configuration")
        return {
            "total_files": len(filtered_files),
            "processed_files": [],
            "failed_files": [{"error": "output.database manquant"}],
            "total_rows_before": 0,
            "total_rows_after": 0,
        }

    driver = db_cfg.get("driver", "postgres")
    if driver != "postgres":
        logger.error(f"Driver non supporté: {driver}")
        return {
            "total_files": len(filtered_files),
            "processed_files": [],
            "failed_files": [{"error": f"Driver non supporté: {driver}"}],
            "total_rows_before": 0,
            "total_rows_after": 0,
        }

    params = db_cfg.get("parameters", {})
    conn_params = {
        "host": params.get("host", "localhost"),
        "port": int(params.get("port", 5432)),
        "dbname": params.get("dbname", "pyjama"),
        "user": params.get("user", "pyjama"),
        "password": params.get("password"),
    }

    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 non installé. Ajoutez psycopg2 ou psycopg[binary] à requirements.txt")
        return {
            "total_files": len(filtered_files),
            "processed_files": [],
            "failed_files": [{"error": "psycopg2 non installé"}],
            "total_rows_before": 0,
            "total_rows_after": 0,
        }

    try:
        conn = psycopg2.connect(**conn_params)
    except Exception as e:
        logger.error(f"Connexion Postgres échouée: {e}")
        return {
            "total_files": len(filtered_files),
            "processed_files": [],
            "failed_files": [{"error": str(e)}],
            "total_rows_before": 0,
            "total_rows_after": 0,
        }
    logger.info("Connexion à la base OK")

    processed_reports: List[Dict[str, Any]] = []
    failed_reports: List[Dict[str, Any]] = []
    total_rows_before = 0
    total_rows_after = 0

    table_name = db_cfg.get("table_name", "").strip()
    if not table_name:
        drawer = (config.get("drawer") or "").strip().upper().replace("-", "_")
        agg = (db_cfg.get("aggregation") or "raw").lower()
        table_name = f"{drawer}_staged_{agg}" if drawer else f"staged_{agg}"

    schema_cfg = db_cfg.get("schema") or {
        "timestamp_column": "time",
        "device_id_column": "device_id",
        "sensor_column": "sensor",
        "value_column": "value",
    }

    aggregation = (db_cfg.get("aggregation") or "raw").lower()
    agg_method = (db_cfg.get("aggregation_method") or "average").lower()
    auto_mode_raw = db_cfg.get("auto_mode", False)
    auto_mode = str(auto_mode_raw).lower() if isinstance(auto_mode_raw, str) else auto_mode_raw
    destroy = bool(db_cfg.get("destroy", False))
    policy = db_cfg.get("policy", "replace")
    exclude_columns = db_cfg.get("exclude_columns") or []
    include_sensors = db_cfg.get("include_sensors")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Liste des tables: {len(tables)} table(s)")

        safe_table = f'"{table_name}"' if not table_name.islower() else table_name
        if destroy:
            logger.info(f"Suppression de la table {table_name} (destroy=True)...")
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {safe_table}")
                conn.commit()
            logger.info("Table supprimée OK")

        _raw81.ensure_table_raw(conn, table_name, schema=schema_cfg)
        logger.info(f"Création/évolution de la table {table_name} OK")

        ts_db_col = schema_cfg.get("timestamp_column", "time")
        max_time = None
        max_time_str: Optional[str] = None  # ISO UTC Z pour comparaison avec colonne Time (Utf8)
        if auto_mode == "max_ts":
            with conn.cursor() as cur:
                cur.execute(f"SELECT MAX({ts_db_col}) FROM {safe_table}")
                max_time = cur.fetchone()[0]
            logger.info(f"Mode auto=max_ts: MAX({ts_db_col}) en base = {max_time}")
            if max_time is not None and getattr(max_time, "tzinfo", None) is not None:
                max_time_str = max_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            elif max_time is not None:
                max_time_str = max_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if max_time is not None:
                filtered = []
                for f in filtered_files:
                    _, last_ts = _raw81.get_file_timestamp_range_raw(f, timestamp_column=ts_col)
                    if last_ts is not None and last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=timezone.utc)
                    if last_ts is None or last_ts > max_time:
                        filtered.append(f)
                logger.info(f"Mode auto=max_ts: {len(filtered)}/{len(filtered_files)} fichiers retenus")
                filtered_files = filtered

        elif auto_mode:
            file_days = _raw81._infer_days_from_files(filtered_files, timestamp_column=ts_col)
            filtered_files = _raw81._auto_mode_filter_files_by_missing_days(
                conn, table_name, ts_db_col, filtered_files, file_days
            )
            if not filtered_files:
                conn.close()
                return {
                    "total_files": len(all_files),
                    "processed_files": [],
                    "failed_files": [],
                    "total_rows_before": 0,
                    "total_rows_after": 0,
                }

        frames: List[pl.DataFrame] = []

        for f in filtered_files:
            report = {"input_file": str(f), "rows_before": 0, "rows_after": 0, "error": None}
            try:
                df = pl.read_parquet(f)
                report["rows_before"] = df.height

                ts_for_df = ts_col
                if ts_for_df not in df.columns:
                    lowered = {c.strip().lower(): c for c in df.columns}
                    key = ts_for_df.strip().lower()
                    if key in lowered:
                        ts_for_df = lowered[key]
                    else:
                        if not df.columns:
                            raise ValueError(f"Aucune colonne lisible dans {f.name}")
                        ts_for_df = df.columns[0]
                        logger.warning(
                            f"Colonne timestamp '{ts_col}' absente dans {f.name}, "
                            f"fallback sur '{ts_for_df}'."
                        )

                df = format_timestamp_column_utc_z(df, ts_for_df)

                if "device_id" not in df.columns:
                    device_id = get_device_id_from_stem(f.stem) or ""
                    df = df.with_columns(pl.lit(device_id).alias("device_id"))

                long_df = _raw81._raw_wide_to_long(
                    df,
                    time_col=ts_for_df,
                    device_id_col="device_id",
                    exclude_columns=exclude_columns,
                )

                if auto_mode == "max_ts" and max_time_str is not None:
                    long_df = long_df.filter(pl.col("Time") > max_time_str)

                if include_sensors and isinstance(include_sensors, list):
                    include_set = {str(s) for s in include_sensors}
                    long_df = long_df.filter(pl.col("sensor").is_in(include_set))

                if long_df.height == 0:
                    processed_reports.append(report)
                    continue

                if aggregation in ("10s", "60s"):
                    if long_df.schema["Time"] == pl.Utf8:
                        long_df = long_df.with_columns(
                            pl.col("Time")
                            .str.to_datetime(time_zone="UTC", strict=False)
                            .alias("Time_dt")
                        )
                    else:
                        long_df = long_df.with_columns(pl.col("Time").alias("Time_dt"))
                    long_df = long_df.with_columns(
                        pl.col("Time_dt").dt.truncate(aggregation).alias("Time_bucket")
                    )
                    agg_expr = (
                        pl.col("value").median()
                        if agg_method == "median"
                        else pl.col("value").mean()
                    )
                    long_df = (
                        long_df.group_by(["Time_bucket", "device_id", "sensor"])
                        .agg(agg_expr.alias("value"))
                        .rename({"Time_bucket": "Time"})
                    )

                value_dtype = long_df.schema.get("value")
                if long_df.height > 0 and "value" in long_df.columns and value_dtype != pl.Null:
                    long_df = long_df.with_columns(
                        pl.col("value").cast(pl.Float64).round(2).alias("value")
                    )

                report["rows_after"] = long_df.height
                frames.append(long_df)
                processed_reports.append(report)
                total_rows_before += report["rows_before"]
                total_rows_after += report["rows_after"]

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {f}: {e}", exc_info=True)
                report["error"] = str(e)
                failed_reports.append(report)

        if not frames:
            logger.warning("Aucune donnée à insérer (tous les fichiers vides ou en échec)")
            conn.close()
            return {
                "total_files": len(filtered_files),
                "processed_files": processed_reports,
                "failed_files": failed_reports,
                "total_rows_before": total_rows_before,
                "total_rows_after": 0,
            }

        combined = pl.concat(frames)

        limit_rows = db_cfg.get("limit_rows")
        if isinstance(limit_rows, int) and limit_rows > 0:
            combined = combined.head(limit_rows)
            logger.info(f"Mode test: limit_rows={limit_rows}")

        if policy == "replace":
            logger.info("Policy replace: suppression des lignes dans la plage concernée...")
            if from_date or to_date:
                def parse_for_db(s: str, is_end: bool = False) -> Optional[str]:
                    if not s or s == "yyyy-mm-ddTh:m:sZ":
                        return None
                    if len(s) == 10 and s.count("-") == 2:
                        s = f"{s}T23:59:59Z" if is_end else f"{s}T00:00:00Z"
                    return s
                min_ts_str = parse_for_db(from_date, is_end=False)
                max_ts_str = parse_for_db(to_date, is_end=True)
            else:
                ts_min = combined["Time"].min()
                ts_max = combined["Time"].max()
                min_ts_str = str(ts_min) if ts_min is not None else None
                max_ts_str = str(ts_max) if ts_max is not None else None
            if min_ts_str and max_ts_str:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {safe_table} "
                        f"WHERE {ts_db_col} >= %s AND {ts_db_col} <= %s",
                        (min_ts_str, max_ts_str),
                    )
                    logger.info(f"Policy replace OK: {cur.rowcount} lignes supprimées")
        elif policy == "replace_all":
            logger.info("Policy replace_all: TRUNCATE...")
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {safe_table}")
        conn.commit()

        n_rows = combined.height
        logger.info(f"Insertion en cours: {n_rows} lignes...")
        inserted = _raw81.insert_rows(
            conn,
            table_name,
            combined,
            schema=schema_cfg,
            commit_each_batch=bool(db_cfg.get("commit_each_batch", False)),
            log_each_batch=bool(db_cfg.get("log_each_batch", False)),
            insert_method=db_cfg.get("insert_method", "executemany"),
            page_size=db_cfg.get("page_size"),
        )
        logger.info(f"Insertion OK: {inserted} lignes dans {table_name}")

    finally:
        conn.close()

    summary = {
        "total_files": len(filtered_files),
        "processed_files": processed_reports,
        "failed_files": failed_reports,
        "total_rows_before": total_rows_before,
        "total_rows_after": total_rows_after,
    }
    logger.info(
        f"81_import_staged_postgres terminé: {len(processed_reports)} traités, "
        f"{len(failed_reports)} échecs, {total_rows_after} lignes après traitement"
    )
    return summary
