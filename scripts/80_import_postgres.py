"""
Script 80_import_postgres : 70_aggregated (Parquet wide) → table PostgreSQL (long) pour PyJAMA.
Lit les Parquet 70_aggregated, convertit en format long (ts, device_id, domain, sensor, value),
crée la table si besoin (autocreate), applique policy replace (delete plage + insert), index pour Grafana.
"""

from pathlib import Path
import sys
import os
import re
from datetime import datetime

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from typing import Any, Dict, List, Optional, Tuple

import logging
from fnmatch import fnmatch

import polars as pl

from format_ts import format_timestamp_column_utc_z
from device_id_helper import get_device_id_from_stem

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Fichier de log explicite (classique) pour 80_import_postgres
DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "debug.log"
FILE_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "80_import_postgres.log"


def _file_log_path_from_config(config: Dict) -> Path:
    """Chemin du fichier de log : output.log_file ou défaut logs/80_import_postgres.log."""
    out = config.get("output", {})
    db = out.get("database", {})
    p = out.get("log_file") or db.get("log_file")
    if p:
        path = Path(p)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return FILE_LOG_PATH


def _ensure_file_logging(config: Dict) -> None:
    """Ajoute un FileHandler au logger racine du script pour écrire dans un fichier de log classique."""
    log_path = _file_log_path_from_config(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(h.baseFilename == str(log_path) for h in logger.handlers if getattr(h, "baseFilename", None)):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)


def get_domain_from_path(path: Path) -> Optional[str]:
    """Extrait le domaine du chemin (ex: .../domain=bio_signal/file.parquet -> bio_signal)."""
    for parent in path.parents:
        if parent.name.startswith("domain="):
            return parent.name.split("=", 1)[1]
    return None


def get_aggregation_from_stem(stem: str) -> Optional[str]:
    """
    Extrait l'agrégation du nom de fichier (ex: ..._agg10s_... -> 10s, ..._agg60s_... -> 60s).
    Pattern: {EXPERIENCE}_{NODE}_{DATA_DAY}_{SUFFIX}_{FILE_TS}, SUFFIX = agg10s, agg60s, etc.
    """
    match = re.search(r"_agg(\d+)s_", stem, re.IGNORECASE)
    if match:
        return match.group(1) + "s"
    match = re.search(r"_agg(\d+)m_", stem, re.IGNORECASE)
    if match:
        return match.group(1) + "m"
    match = re.search(r"_agg(\d+)h_", stem, re.IGNORECASE)
    if match:
        return match.group(1) + "h"
    match = re.search(r"_agg(\d+)d_", stem, re.IGNORECASE)
    if match:
        return match.group(1) + "d"
    return None


def get_file_timestamp_range(
    path: Path, timestamp_column: str = "ts"
) -> Tuple[Optional[datetime], Optional[datetime]]:
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
    return None, None


def wide_to_long(
    df: pl.DataFrame,
    ts_col: str,
    id_vars: List[str],
    column_to_sensor: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Convertit un DataFrame wide (ts, device_id, domain, c0, c1, ...) en long (ts, device_id, domain, sensor, value).
    column_to_sensor optionnel : {"c0": "m0", "c1": "m1"} pour renommer les capteurs.
    """
    value_vars = [c for c in df.columns if c not in id_vars]
    if not value_vars:
        return df.select(id_vars).with_columns(
            pl.lit(None).cast(pl.Utf8).alias("sensor"),
            pl.lit(None).cast(pl.Float64).alias("value"),
        )
    molten = df.select(id_vars + value_vars).melt(
        id_vars=id_vars,
        value_vars=value_vars,
        variable_name="sensor",
        value_name="value",
    )
    if column_to_sensor:
        molten = molten.with_columns(
            pl.col("sensor").replace(column_to_sensor).alias("sensor")
        )
    return molten


def infer_value_type(df: pl.DataFrame, value_col: str = "value") -> Tuple[str, Optional[int], Optional[int]]:
    """
    Infère le type SQL pour la colonne value : INTEGER ou NUMERIC(precision, scale).
    Retourne ("int", None, None) ou ("numeric", precision, scale).
    """
    if value_col not in df.columns:
        return "numeric", 18, 6
    col = df[value_col]
    non_null = col.drop_nulls()
    if non_null.len() == 0:
        return "numeric", 18, 6
    if col.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "int", None, None
    if col.dtype in (pl.Float32, pl.Float64):
        try:
            cast_int = non_null.cast(pl.Int64).cast(pl.Float64)
            diff = (non_null - cast_int).abs()
            if diff.max() is not None and float(diff.max()) < 1e-9:
                return "int", None, None
        except Exception:
            pass
    return "numeric", 18, 6


def get_table_name(
    config: Dict,
    domain: Optional[str],
    aggregation: Optional[str],
    first_file: Optional[Path],
) -> str:
    """
    Retourne le nom de table : explicite dans output.database.table_name,
    ou dérivé {drawer}_{domain}_{aggregation} si absent.
    """
    db_cfg = config.get("output", {}).get("database", {})
    explicit = db_cfg.get("table_name", "").strip()
    if explicit:
        return explicit
    drawer = (config.get("drawer") or "").strip().upper().replace("-", "_")
    if not drawer and first_file:
        # Essayer d'extraire experience du nom de fichier (premanip-grace_... -> PREMANIP_GRACE)
        stem = first_file.stem
        if "_" in stem:
            drawer = stem.split("_")[0].upper().replace("-", "_")
    domain_part = (domain or "data").lower().replace("-", "_")
    agg_part = (aggregation or "raw").lower()
    return f"{drawer}_{domain_part}_{agg_part}"


def ensure_table(
    conn: Any,
    table_name: str,
    value_sql_type: str,
    value_precision: Optional[int] = None,
    value_scale: Optional[int] = None,
    ts_type: str = "TIMESTAMPTZ",
    schema: Optional[Dict[str, str]] = None,
    indexes: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Crée la table si elle n'existe pas.
    Schéma par défaut : ts, device_id, domain, sensor, value.
    Index par défaut : (ts), (device_id, ts).
    """
    value_col_def = "INTEGER"
    if value_sql_type == "numeric" and value_precision is not None and value_scale is not None:
        value_col_def = f"NUMERIC({value_precision},{value_scale})"
    elif value_sql_type == "numeric":
        value_col_def = "NUMERIC(18,6)"

    # Colonnes de la table (schéma configurable)
    schema = schema or {}
    ts_col = schema.get("timestamp_column", "ts")
    device_id_col = schema.get("device_id_column", "device_id")
    domain_col = schema.get("domain_column", "domain")
    sensor_col = schema.get("sensor_column", "sensor")
    value_col = schema.get("value_column", "value")

    # Identifier PostgreSQL : noms entre guillemets pour préserver la casse si besoin
    safe_table = f'"{table_name}"' if not table_name.islower() else table_name
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {safe_table} (
        {ts_col} {ts_type} NOT NULL,
        {device_id_col} VARCHAR(255) NOT NULL,
        {domain_col} VARCHAR(255) NOT NULL,
        {sensor_col} VARCHAR(255) NOT NULL,
        {value_col} {value_col_def}
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
    conn.commit()

    # Index pour requêtes temps + device (Grafana), configurables
    if indexes is None:
        # Défaut : index sur ts et sur (device_id, ts)
        default_indexes = [
            {
                "name": f"{table_name}_idx_ts",
                "columns": [ts_col],
                "method": "btree",
                "unique": False,
            },
            {
                "name": f"{table_name}_idx_device_ts",
                "columns": [device_id_col, ts_col],
                "method": "btree",
                "unique": False,
            },
        ]
    else:
        default_indexes = indexes

    with conn.cursor() as cur:
        for idx in default_indexes:
            name = idx.get("name")
            cols = idx.get("columns") or []
            method = idx.get("method", "btree")
            unique = bool(idx.get("unique", False))
            if not name or not cols:
                continue

            safe_idx = f'"{name}"' if not name.islower() else name
            cols_sql = ", ".join(cols)
            unique_sql = "UNIQUE " if unique else ""
            cur.execute(
                f"CREATE {unique_sql}INDEX IF NOT EXISTS {safe_idx} "
                f"ON {safe_table} USING {method} ({cols_sql});"
            )
    conn.commit()
    logger.info(f"Table {table_name} prête (index ts, device_id+ts)")


def delete_replace_range(conn: Any, table_name: str, ts_column: str, min_ts: str, max_ts: str) -> int:
    """Supprime les lignes dont ts_column est dans [min_ts, max_ts]. Retourne le nombre de lignes supprimées."""
    safe_table = f'"{table_name}"' if not table_name.islower() else table_name
    safe_ts_col = f'"{ts_column}"' if not ts_column.islower() else ts_column
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {safe_table} WHERE {safe_ts_col} >= %s AND {safe_ts_col} <= %s",
            (min_ts, max_ts),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted or 0


BATCH_INSERT_SIZE = 10_000


def insert_rows(conn: Any, table_name: str, df: pl.DataFrame, schema: Optional[Dict[str, str]] = None) -> int:
    """Insère les lignes du DataFrame (colonnes internes ts, device_id, domain, sensor, value) dans la table cible.
    Le mapping vers les noms de colonnes SQL est configurable via schema.
    Insertion par lots pour éviter blocage apparent et limiter la charge mémoire.
    """
    if df.height == 0:
        return 0
    safe_table = f'"{table_name}"' if not table_name.islower() else table_name

    schema = schema or {}
    ts_col = schema.get("timestamp_column", "ts")
    device_id_col = schema.get("device_id_column", "device_id")
    domain_col = schema.get("domain_column", "domain")
    sensor_col = schema.get("sensor_column", "sensor")
    value_col = schema.get("value_column", "value")

    rows = df.to_dicts()
    insert_sql = (
        f"INSERT INTO {safe_table} ({ts_col}, {device_id_col}, {domain_col}, {sensor_col}, {value_col}) "
        f"VALUES (%s, %s, %s, %s, %s)"
    )

    with conn.cursor() as cur:
        total = 0
        for i in range(0, len(rows), BATCH_INSERT_SIZE):
            batch = rows[i : i + BATCH_INSERT_SIZE]
            cur.executemany(
                insert_sql,
                [
                    (
                        r.get("ts"),
                        str(r.get("device_id", "")),
                        str(r.get("domain", "")),
                        str(r.get("sensor", "")),
                        r.get("value"),
                    )
                    for r in batch
                ],
            )
            total += len(batch)
            if total % (10 * BATCH_INSERT_SIZE) == 0 and total > 0:
                logger.info(f"  Insertion: {total}/{len(rows)} lignes...")
    conn.commit()
    return len(rows)


def process_single_file(
    input_path: Path, config: Dict, ts_col: str, column_to_sensor: Optional[Dict[str, str]]
) -> Tuple[Optional[pl.DataFrame], Dict]:
    """
    Lit un Parquet 70_aggregated (wide), convertit en long. Retourne (DataFrame long, rapport).
    Si device_id ou domain sont absents (ex: output_columns 70 les a exclus), les infère depuis le chemin et le nom de fichier.
    """
    report = {
        "input_file": str(input_path),
        "rows_before": 0,
        "rows_after": 0,
        "error": None,
    }
    try:
        df = pl.read_parquet(input_path)
    except Exception as e:
        report["error"] = str(e)
        return None, report

    # #region agent log
    try:
        import json
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({"hypothesisId": "H1-H5", "location": "80_import_postgres.py:process_single_file", "message": "columns_after_read", "data": {"path": str(input_path), "columns": list(df.columns)}, "timestamp": int(datetime.utcnow().timestamp() * 1000), "sessionId": "debug-session"}) + "\n")
    except Exception:
        pass
    # #endregion

    if ts_col not in df.columns:
        report["error"] = f"Colonne manquante: {ts_col}"
        return None, report

    domain = get_domain_from_path(input_path)
    device_id = get_device_id_from_stem(input_path.stem)
    if "device_id" not in df.columns and device_id:
        df = df.with_columns(pl.lit(device_id).alias("device_id"))
    elif "device_id" not in df.columns:
        df = df.with_columns(pl.lit("").alias("device_id"))
    if "domain" not in df.columns and domain:
        df = df.with_columns(pl.lit(domain).alias("domain"))
    elif "domain" not in df.columns:
        df = df.with_columns(pl.lit("").alias("domain"))

    id_vars = [ts_col, "device_id", "domain"]
    value_vars = [c for c in df.columns if c not in id_vars]

    # Filtrage optionnel des métriques à importer depuis la config 80
    # (output.database.include_metrics = ["m0", "m1", ...]).
    db_cfg = config.get("output", {}).get("database", {})
    include_metrics = db_cfg.get("include_metrics")
    if isinstance(include_metrics, list) and include_metrics:
        include_set = {str(m) for m in include_metrics}
        value_vars = [c for c in value_vars if c in include_set]

    # #region agent log
    try:
        import json
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "hypothesisId": "H3-H5",
                        "location": "80_import_postgres.py:process_single_file",
                        "message": "id_vars_value_vars",
                        "data": {
                            "path": str(input_path),
                            "columns": list(df.columns),
                            "id_vars": id_vars,
                            "value_vars": value_vars,
                            "include_metrics": include_metrics,
                        },
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "sessionId": "debug-session",
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    if not value_vars:
        msg = "Aucune colonne de valeur (ex. c0, c1, ...) dans le Parquet 70_aggregated"
        report["error"] = msg
        logger.error(
            "%s | path=%s | colonnes_lues=%s | id_vars=%s | value_vars=%s | cause_probable=Parquet_70_ne_contient_que_ts_ou_ts_device_id_domain_verifier_70_output_columns_et_metrics_aggregation",
            msg, input_path, list(df.columns), id_vars, value_vars,
        )
        # #region agent log
        try:
            import json
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"hypothesisId": "H1", "location": "80_import_postgres.py:process_single_file", "message": "skip_no_value_columns", "data": {"path": str(input_path), "columns": list(df.columns), "id_vars": id_vars, "value_vars": value_vars}, "timestamp": int(datetime.utcnow().timestamp() * 1000), "sessionId": "debug-session"}) + "\n")
        except Exception:
            pass
        # #endregion
        return None, report

    long_df = wide_to_long(df, ts_col, id_vars, column_to_sensor)
    report["rows_before"] = df.height
    report["rows_after"] = long_df.height

    # Normaliser ts au format UTC Z
    long_df = format_timestamp_column_utc_z(long_df, "ts")
    return long_df, report


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 70_aggregated (Parquet) → table PostgreSQL (long).
    Retourne un résumé compatible pyjama (total_files, processed_files, failed_files, total_rows_*).
    """
    _ensure_file_logging(config)
    logger.info("Début 80_import_postgres (70_aggregated → PostgreSQL)")

    input_cfg = config.get("input", {})
    input_dir = Path(input_cfg["input_directory"])
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    file_pattern = input_cfg.get("file_pattern", "*.parquet")
    search_subdirs = input_cfg.get("search_in_subdirectory", "no").lower() == "yes"
    except_pattern = input_cfg.get("except", "") or ""
    ts_col = input_cfg.get("timestamp_column", "ts")

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
    # Mot de passe uniquement depuis la configuration JSON (parameters.password)
    password = params.get("password")
    conn_params = {
        "host": params.get("host", "localhost"),
        "port": int(params.get("port", 5432)),
        "dbname": params.get("dbname", "pyjama"),
        "user": params.get("user", "pyjama"),
        "password": password,
    }
    if not conn_params["password"]:
        logger.warning("Mot de passe Postgres non fourni dans output.database.parameters.password")

    column_to_sensor = config.get("output", {}).get("database", {}).get("column_to_sensor")
    if not isinstance(column_to_sensor, dict):
        column_to_sensor = None

    # Lire tous les fichiers et concaténer en long
    first_file = filtered_files[0]
    domain = get_domain_from_path(first_file)
    # Agrégation : config output.database.aggregation prioritaire, sinon déduite du nom du premier fichier
    aggregation = (db_cfg.get("aggregation") or "").strip() or get_aggregation_from_stem(first_file.stem)
    table_name = get_table_name(config, domain, aggregation, first_file)
    logger.info(f"Table cible: {table_name} (domain={domain}, aggregation={aggregation})")

    frames: List[pl.DataFrame] = []
    total_rows_before = 0
    processed_reports = []
    failed_reports = []

    for file_path in filtered_files:
        long_df, report = process_single_file(file_path, config, ts_col, column_to_sensor)
        if report.get("error"):
            failed_reports.append(report)
            continue
        processed_reports.append(report)
        total_rows_before += report.get("rows_before", 0)
        if long_df is not None and long_df.height > 0:
            frames.append(long_df)

    if not frames:
        logger.warning("Aucune donnée à insérer (tous les fichiers vides ou en échec)")
        return {
            "total_files": len(filtered_files),
            "processed_files": processed_reports,
            "failed_files": failed_reports,
            "total_rows_before": total_rows_before,
            "total_rows_after": 0,
        }

    combined = pl.concat(frames)
    total_rows_after = combined.height

    # Inférer le type value pour la table (modifiable par config)
    value_type, value_prec, value_scale = infer_value_type(combined, "value")

    # Surcharges éventuelles depuis la configuration :
    # - value_type: "int" / "integer" / "numeric"
    # - value_precision, value_scale: pour NUMERIC(p,s)
    cfg_value_type = db_cfg.get("value_type")
    if isinstance(cfg_value_type, str):
        cfg_value_type_lower = cfg_value_type.strip().lower()
        if cfg_value_type_lower in ("int", "integer"):
            value_type, value_prec, value_scale = "int", None, None
        elif cfg_value_type_lower == "numeric":
            value_type = "numeric"
            if "value_precision" in db_cfg and "value_scale" in db_cfg:
                try:
                    value_prec = int(db_cfg["value_precision"])
                    value_scale = int(db_cfg["value_scale"])
                except (TypeError, ValueError):
                    pass

    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 non installé. Ajoutez psycopg2 ou psycopg[binary] à requirements.txt")
        return {
            "total_files": len(filtered_files),
            "processed_files": processed_reports,
            "failed_files": failed_reports + [{"error": "psycopg2 non installé"}],
            "total_rows_before": total_rows_before,
            "total_rows_after": 0,
        }

    try:
        conn = psycopg2.connect(**conn_params)
    except Exception as e:
        logger.error(f"Connexion Postgres échouée: {e}")
        return {
            "total_files": len(filtered_files),
            "processed_files": processed_reports,
            "failed_files": failed_reports + [{"error": str(e)}],
            "total_rows_before": total_rows_before,
            "total_rows_after": 0,
        }

    try:
        autocreate = db_cfg.get("autocreate", True)
        destroy = bool(db_cfg.get("destroy", False))

        # Schéma et index configurables pour la table cible
        schema_cfg = db_cfg.get("schema") or {}
        indexes_cfg = db_cfg.get("indexes")

        # Option destroy: drop complet de la table, puis recréation
        if destroy:
            safe_table = f'"{table_name}"' if not table_name.islower() else table_name
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {safe_table}")
            conn.commit()
            logger.info(f"Option destroy=True : table {table_name} supprimée avant recréation")

            # En mode destroy, on force la recréation même si autocreate est False
            ensure_table(
                conn,
                table_name,
                value_type,
                value_precision=value_prec,
                value_scale=value_scale,
                ts_type="TIMESTAMPTZ",
                schema=schema_cfg,
                indexes=indexes_cfg,
            )
        elif autocreate:
            ensure_table(
                conn,
                table_name,
                value_type,
                value_precision=value_prec,
                value_scale=value_scale,
                ts_type="TIMESTAMPTZ",
                schema=schema_cfg,
                indexes=indexes_cfg,
            )

        policy = db_cfg.get("policy", "replace")
        ts_min = combined["ts"].min()
        ts_max = combined["ts"].max()
        min_ts_str = str(ts_min) if ts_min is not None else None
        max_ts_str = str(ts_max) if ts_max is not None else None

        # Colonne timestamp côté SQL (par défaut "ts")
        ts_db_col = schema_cfg.get("timestamp_column", "ts")

        if policy == "replace" and min_ts_str and max_ts_str:
            deleted = delete_replace_range(conn, table_name, ts_db_col, min_ts_str, max_ts_str)
            logger.info(f"Policy replace: {deleted} lignes supprimées dans [{min_ts_str}, {max_ts_str}]")
        elif policy == "replace_all":
            safe_table = f'"{table_name}"' if not table_name.islower() else table_name
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {safe_table}")
            conn.commit()
            logger.info(f"Policy replace_all: table {table_name} vidée")

        n_rows = combined.height
        logger.info(f"Insertion en cours: {n_rows} lignes (par lots de 10000)...")
        inserted = insert_rows(conn, table_name, combined, schema=schema_cfg)
        logger.info(f"Insertion: {inserted} lignes dans {table_name}")
    finally:
        conn.close()

    summary = {
        "total_files": len(filtered_files),
        "processed_files": processed_reports,
        "failed_files": failed_reports,
        "total_rows_before": total_rows_before,
        "total_rows_after": total_rows_after,
    }

    # #region agent log
    try:
        import json as _json
        from datetime import datetime as _dt

        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H1-H3",
                        "location": "80_import_postgres.py:run:return",
                        "message": "summary_structure",
                        "data": {
                            "total_files": summary.get("total_files"),
                            "processed_count": len(summary.get("processed_files", [])),
                            "failed_count": len(summary.get("failed_files", [])),
                            "failed_files_sample": summary.get("failed_files", [])[:3],
                        },
                        "timestamp": int(_dt.utcnow().timestamp() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    logger.info(
        f"80_import_postgres terminé: {len(processed_reports)} fichiers traités, {len(failed_reports)} échecs, {total_rows_after} lignes en base"
    )
    return summary
