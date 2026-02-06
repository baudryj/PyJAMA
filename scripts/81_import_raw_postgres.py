"""
Script 81_import_raw_postgres : 00_raw (fichiers bruts) → table PostgreSQL (long) pour PyJAMA.

Lit les fichiers raw (CSV/Parquet) dans 00_raw, normalise la colonne temps, infère device_id
depuis le nom de fichier, convertit en format long (Time, device_id, sensor, value), applique
éventuellement une agrégation 10s/60s, puis insère en base Postgres pour visualisation Grafana.
"""

from pathlib import Path
import sys
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Iterable

import logging
from fnmatch import fnmatch

import polars as pl

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from format_ts import format_timestamp_column_utc_z
from device_id_helper import get_device_id_from_stem


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _parse_date_only(s: str) -> Optional[date]:
    """Retourne une date si s est au format yyyy-mm-dd, sinon None."""
    try:
        if len(s) == 10 and s.count("-") == 2:
            return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None
    return None


def get_day_from_stem(stem: str) -> Optional[date]:
    """
    Essaye d'extraire un jour à partir du nom de fichier raw :
    pattern typique : <experience>_<device_id>_<date_day>_...
    """
    parts = stem.split("_")
    for part in parts:
        d = _parse_date_only(part)
        if d is not None:
            return d
    return None


def get_file_timestamp_range_raw(
    path: Path, timestamp_column: str = "Time"
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Lit la plage temporelle (min/max) d'un fichier raw, en fonction de l'extension :
    - Parquet : scan_parquet + min/max sur timestamp_column
    - CSV : lit première et dernière ligne non vide, suppose la colonne temps en première colonne.
    """
    formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"]

    def parse_ts(s: Optional[str]) -> Optional[datetime]:
        if s is None:
            return None
        txt = str(s).strip()
        if not txt:
            return None
        for fmt in formats:
            try:
                return datetime.strptime(txt, fmt)
            except (ValueError, TypeError):
                continue
        return None

    if path.suffix.lower() == ".parquet":
        try:
            row = (
                pl.scan_parquet(path)
                .select(
                    pl.col(timestamp_column).min().alias("min_ts"),
                    pl.col(timestamp_column).max().alias("max_ts"),
                )
                .collect()
            )
            if row.height == 0:
                return None, None
            min_ts_str = row["min_ts"][0]
            max_ts_str = row["max_ts"][0]
            return parse_ts(min_ts_str), parse_ts(max_ts_str)
        except Exception as e:  # pragma: no cover - log de robustesse
            logger.warning(f"Erreur lecture plage temporelle Parquet de {path}: {e}")
            return None, None

    # CSV ou autre : on lit première et dernière ligne non vide
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            first_line = ""
            for line in f:
                line = line.strip()
                if line:
                    first_line = line
                    break
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
    except Exception as e:  # pragma: no cover - log de robustesse
        logger.warning(f"Erreur lecture plage temporelle de {path}: {e}")
        return None, None


def _parse_semicolon_kv_file(path: Path) -> pl.DataFrame:
    """
    Parse un fichier raw au format:
    timestamp;capteur:valeur;capteur:valeur;...
    Best-effort: ignore les lignes invalides.
    """
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 2:
                    continue
                ts = parts[0].strip()
                if not ts:
                    continue
                row: Dict[str, Any] = {"Time": ts}
                ok = False
                for item in parts[1:]:
                    if ":" not in item:
                        continue
                    k, v = item.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if not k:
                        continue
                    try:
                        row[k] = float(v)
                    except Exception:
                        # best-effort: ignore valeur invalide
                        continue
                    ok = True
                if ok:
                    rows.append(row)
    except Exception:
        return pl.DataFrame()
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _raw_wide_to_long(
    df: pl.DataFrame,
    time_col: str,
    device_id_col: str = "device_id",
    exclude_columns: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    """
    Convertit un DataFrame raw wide en long (Time, device_id, sensor, value).
    - time_col : nom de la colonne temps (ex. "Time")
    - device_id_col : nom de la colonne device_id dans df
    - exclude_columns : colonnes supplémentaires à exclure des capteurs.
    """
    exclude = {time_col, device_id_col}
    if exclude_columns:
        exclude.update(exclude_columns)
    id_vars = [c for c in [time_col, device_id_col] if c in df.columns]
    value_vars = [c for c in df.columns if c not in exclude]

    if not value_vars:
        # Aucun capteur exploitable dans ce fichier : on retourne un DataFrame vide
        # avec le schéma attendu (Time, device_id, sensor, value) pour que la suite
        # du pipeline puisse continuer sans erreur.
        return pl.DataFrame(
            {
                "Time": [],
                "device_id": [],
                "sensor": [],
                "value": [],
            }
        )

    molten = df.select(id_vars + value_vars).melt(
        id_vars=id_vars,
        value_vars=value_vars,
        variable_name="sensor",
        value_name="value",
    )
    # Harmoniser le nom de la colonne temps en "Time" pour la suite
    if time_col != "Time" and time_col in molten.columns:
        molten = molten.rename({time_col: "Time"})
    if device_id_col != "device_id" and device_id_col in molten.columns:
        molten = molten.rename({device_id_col: "device_id"})
    return molten


def _infer_days_from_files(
    files: List[Path],
    timestamp_column: str,
) -> Dict[Path, Optional[date]]:
    """
    Retourne un mapping fichier -> jour (date) en essayant d'abord le nom de fichier,
    puis la plage temporelle si nécessaire.
    """
    result: Dict[Path, Optional[date]] = {}
    for f in files:
        d = get_day_from_stem(f.stem)
        if d is not None:
            result[f] = d
            continue
        first_ts, _ = get_file_timestamp_range_raw(f, timestamp_column=timestamp_column)
        result[f] = first_ts.date() if first_ts is not None else None
    return result


def _auto_mode_filter_files_by_missing_days(
    conn: Any,
    table_name: str,
    ts_column_db: str,
    files: List[Path],
    file_days: Dict[Path, Optional[date]],
) -> List[Path]:
    """
    En mode auto, ne conserve que les fichiers dont le jour n'est pas présent dans la table.
    """
    safe_table = f'"{table_name}"' if not table_name.islower() else table_name

    # Collecter les jours connus à partir des fichiers
    candidate_days = sorted({d for d in file_days.values() if d is not None})
    if not candidate_days:
        return files

    # Récupérer les jours déjà présents en base
    placeholders = ", ".join(["%s"] * len(candidate_days))
    query = (
        f"SELECT DISTINCT DATE({ts_column_db}) AS day_present "
        f"FROM {safe_table} WHERE DATE({ts_column_db}) IN ({placeholders})"
    )
    present_days: List[date] = []
    with conn.cursor() as cur:
        cur.execute(query, candidate_days)
        rows = cur.fetchall()
        present_days = [r[0] for r in rows]

    present_set = set(present_days)
    missing_days = {d for d in candidate_days if d not in present_set}
    if not missing_days:
        logger.info("Mode auto: aucun jour manquant détecté, rien à charger.")
        return []

    logger.info(f"Mode auto: jours manquants détectés: {sorted(missing_days)}")
    filtered = [f for f in files if file_days.get(f) in missing_days]
    logger.info(f"Mode auto: {len(filtered)}/{len(files)} fichiers retenus pour complétion des jours manquants")
    return filtered


BATCH_INSERT_SIZE = 10_000


def insert_rows(
    conn: Any,
    table_name: str,
    df: pl.DataFrame,
    schema: Optional[Dict[str, str]] = None,
    commit_each_batch: bool = False,
    log_each_batch: bool = False,
    insert_method: str = "executemany",
    page_size: Optional[int] = None,
) -> int:
    """
    Insère les lignes du DataFrame (Time, device_id, sensor, value) dans la table cible.
    Le mapping vers les noms de colonnes SQL est configurable via schema.
    Insertion par lots pour performance.
    """
    if df.height == 0:
        return 0

    safe_table = f'"{table_name}"' if not table_name.islower() else table_name

    schema = schema or {}
    ts_col = schema.get("timestamp_column", "Time")
    device_id_col = schema.get("device_id_column", "device_id")
    sensor_col = schema.get("sensor_column", "sensor")
    value_col = schema.get("value_column", "value")

    rows = df.to_dicts()
    insert_sql = (
        f"INSERT INTO {safe_table} ({ts_col}, {device_id_col}, {sensor_col}, {value_col}) "
        f"VALUES (%s, %s, %s, %s)"
    )
    insert_sql_values = (
        f"INSERT INTO {safe_table} ({ts_col}, {device_id_col}, {sensor_col}, {value_col}) "
        f"VALUES %s"
    )
    method = (insert_method or "executemany").lower()
    batch_size = page_size or BATCH_INSERT_SIZE

    # #region agent log
    try:
        import json
        from pathlib import Path as _Path
        _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
        with open(_dbg, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H_OPT_1",
                        "location": "81_import_raw_postgres.py:insert_rows:start",
                        "message": "insert_start",
                        "data": {
                            "rows": len(rows),
                            "method": method,
                            "batch_size": batch_size,
                        },
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    if method not in ("executemany", "execute_values"):
        method = "executemany"

    with conn.cursor() as cur:
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_tuples = [
                (
                    r.get("Time"),
                    str(r.get("device_id", "")),
                    str(r.get("sensor", "")),
                    r.get("value"),
                )
                for r in batch
            ]

            import time as _time
            t0 = _time.monotonic()
            if method == "execute_values":
                from psycopg2.extras import execute_values
                execute_values(cur, insert_sql_values, batch_tuples, page_size=len(batch_tuples))
            else:
                cur.executemany(insert_sql, batch_tuples)
            t1 = _time.monotonic()

            total += len(batch)
            if log_each_batch:
                logger.info(f"  Batch insert: {total}/{len(rows)} lignes ({t1 - t0:.2f}s)")
            elif total % (10 * BATCH_INSERT_SIZE) == 0 and total > 0:
                logger.info(f"  Insertion: {total}/{len(rows)} lignes...")
            if commit_each_batch:
                conn.commit()
                if log_each_batch:
                    logger.info("  Batch commit OK")

            # #region agent log
            try:
                import json
                from pathlib import Path as _Path
                _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                with open(_dbg, "a", encoding="utf-8") as _f:
                    _f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H_OPT_2",
                                "location": "81_import_raw_postgres.py:insert_rows:batch",
                                "message": "batch_done",
                                "data": {
                                    "batch_rows": len(batch),
                                    "total": total,
                                    "elapsed_s": round(t1 - t0, 4),
                                    "method": method,
                                },
                                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
    conn.commit()

    # #region agent log
    try:
        import json
        from pathlib import Path as _Path
        _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
        with open(_dbg, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "H_OPT_3",
                        "location": "81_import_raw_postgres.py:insert_rows:end",
                        "message": "insert_done",
                        "data": {
                            "rows": len(rows),
                            "method": method,
                            "batch_size": batch_size,
                        },
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    return len(rows)


def ensure_table_raw(
    conn: Any,
    table_name: str,
    schema: Optional[Dict[str, str]] = None,
    numeric_precision: Optional[int] = 18,
    numeric_scale: Optional[int] = 2,
) -> None:
    """
    Crée la table si elle n'existe pas, ou ajoute les colonnes manquantes :
    - Time TIMESTAMPTZ
    - device_id TEXT
    - sensor TEXT
    - value NUMERIC(p,s) ou DOUBLE PRECISION
    """
    schema = schema or {}
    ts_col = schema.get("timestamp_column", "Time")
    device_id_col = schema.get("device_id_column", "device_id")
    sensor_col = schema.get("sensor_column", "sensor")
    value_col = schema.get("value_column", "value")

    safe_table = f'"{table_name}"' if not table_name.islower() else table_name

    with conn.cursor() as cur:
        # Créer la table si elle n'existe pas
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {safe_table} (
                {ts_col} TIMESTAMPTZ,
                {device_id_col} TEXT,
                {sensor_col} TEXT,
                {value_col} NUMERIC({numeric_precision},{numeric_scale})
            )
            """
        )
        # Récupérer les colonnes existantes
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            """,
            (table_name,),
        )
        # Postgres stocke les noms non quotés en minuscules → comparaison insensible à la casse.
        existing_cols = {r[0].lower() for r in cur.fetchall()}
        ts_col_l = ts_col.lower()
        device_id_col_l = device_id_col.lower()
        sensor_col_l = sensor_col.lower()
        value_col_l = value_col.lower()

        # Ajouter les colonnes manquantes si nécessaire (en tenant compte de la casse)
        if ts_col_l not in existing_cols:
            cur.execute(f"ALTER TABLE {safe_table} ADD COLUMN {ts_col} TIMESTAMPTZ")
        if device_id_col_l not in existing_cols:
            cur.execute(f"ALTER TABLE {safe_table} ADD COLUMN {device_id_col} TEXT")
        if sensor_col_l not in existing_cols:
            cur.execute(f"ALTER TABLE {safe_table} ADD COLUMN {sensor_col} TEXT")
        if value_col_l not in existing_cols:
            cur.execute(
                f"ALTER TABLE {safe_table} ADD COLUMN {value_col} "
                f"NUMERIC({numeric_precision},{numeric_scale})"
            )

        # Index temps et (device_id, temps)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name.lower()}_idx_time "
            f"ON {safe_table} ({ts_col})"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name.lower()}_idx_device_time "
            f"ON {safe_table} ({device_id_col}, {ts_col})"
        )

    conn.commit()


def _apply_time_filters_to_files(
    files: List[Path],
    from_date: str,
    to_date: str,
    timestamp_column: str,
) -> List[Path]:
    """Filtre les fichiers selon une plage from/to.

    Priorité : date dans le nom de fichier (YYYY-MM-DD) si disponible,
    sinon fallback sur min/max de la colonne temps.
    """

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

    if not from_dt and not to_dt:
        return files

    filtered: List[Path] = []
    for f in files:
        # 1) Filtrage par date dans le nom de fichier si possible
        file_day = get_day_from_stem(f.stem)
        if file_day is not None and (from_dt or to_dt):
            include = True
            if from_dt and file_day < from_dt.date():
                include = False
            if to_dt and file_day > to_dt.date():
                include = False
            if include:
                filtered.append(f)
            else:
                logger.info(f"Fichier {f.name} exclu (hors plage temporelle via nom)")
            continue

        # 2) Fallback : lecture min/max sur la colonne temps
        first_ts, last_ts = get_file_timestamp_range_raw(f, timestamp_column=timestamp_column)
        if first_ts is None or last_ts is None:
            logger.warning(f"Plage temporelle illisible pour {f}, fichier inclus")
            filtered.append(f)
            continue
        include = True
        if from_dt and last_ts < from_dt:
            include = False
        if to_dt and first_ts > to_dt:
            include = False
        if include:
            filtered.append(f)
        else:
            logger.info(f"Fichier {f.name} exclu (hors plage temporelle)")
    logger.info(f"{len(filtered)} fichiers après filtrage temporel")
    return filtered


def run(config: Dict) -> Dict:
    """
    Point d'entrée : 00_raw (CSV/Parquet) → table PostgreSQL (long).
    Retourne un résumé compatible pyjama (total_files, processed_files, failed_files, total_rows_*).
    """
    logger.info("Début 81_import_raw_postgres (00_raw → PostgreSQL)")

    input_cfg = config.get("input", {})
    input_dir = Path(input_cfg["input_directory"])
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    file_pattern = input_cfg.get("file_pattern", "*")
    search_subdirs = input_cfg.get("search_in_subdirectory", "no").lower() == "yes"
    except_pattern = input_cfg.get("except", "") or ""
    ts_col = input_cfg.get("timestamp_column", "Time")

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
        filtered_files = _apply_time_filters_to_files(filtered_files, from_date, to_date, ts_col)

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
    password = params.get("password")
    conn_params = {
        "host": params.get("host", "localhost"),
        "port": int(params.get("port", 5432)),
        "dbname": params.get("dbname", "pyjama"),
        "user": params.get("user", "pyjama"),
        "password": password,
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
        # Fallback simple : drawer_raw_<aggregation>
        drawer = (config.get("drawer") or "").strip().upper().replace("-", "_")
        agg = (db_cfg.get("aggregation") or "raw").lower()
        table_name = f"{drawer}_raw_{agg}" if drawer else f"raw_{agg}"

    schema_cfg = db_cfg.get("schema") or {
        "timestamp_column": "Time",
        "device_id_column": "device_id",
        "sensor_column": "sensor",
        "value_column": "value",
    }

    aggregation = (db_cfg.get("aggregation") or "raw").lower()
    agg_method = (db_cfg.get("aggregation_method") or "average").lower()
    auto_mode = bool(db_cfg.get("auto_mode", False))
    destroy = bool(db_cfg.get("destroy", False))
    policy = db_cfg.get("policy", "replace")

    try:
        # Étape 1 : liste des tables (validation)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Liste des tables (schéma courant): {len(tables)} table(s) — {tables[:10]}{'...' if len(tables) > 10 else ''}")

        # Étape 2 : suppression de la table si destroy=True
        safe_table = f'"{table_name}"' if not table_name.islower() else table_name
        if destroy:
            logger.info(f"Suppression de la table {table_name} (destroy=True)...")
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {safe_table}")
                conn.commit()
            logger.info("Table supprimée OK")

        # Étape 3 : création / évolution de la table
        ensure_table_raw(conn, table_name, schema=schema_cfg)
        logger.info(f"Création/évolution de la table {table_name} OK")

        # Mode auto : ne charger que les jours manquants
        file_days = _infer_days_from_files(filtered_files, timestamp_column=ts_col)
        if auto_mode:
            ts_db_col = schema_cfg.get("timestamp_column", "Time")
            filtered_files = _auto_mode_filter_files_by_missing_days(
                conn, table_name, ts_db_col, filtered_files, file_days
            )
            if not filtered_files:
                return {
                    "total_files": len(all_files),
                    "processed_files": [],
                    "failed_files": [],
                    "total_rows_before": 0,
                    "total_rows_after": 0,
                }

        # Lecture des fichiers et construction du DataFrame long combiné
        frames: List[pl.DataFrame] = []
        include_sensors = db_cfg.get("include_sensors")
        exclude_columns = db_cfg.get("exclude_columns") or []

        for f in filtered_files:
            report = {
                "input_file": str(f),
                "rows_before": 0,
                "rows_after": 0,
                "error": None,
            }
            try:
                if f.suffix.lower() == ".parquet":
                    df = pl.read_parquet(f)
                else:
                    df = pl.read_csv(f, null_values="NaN")
                report["rows_before"] = df.height

                # Normaliser la colonne temps (approche tolérante : casse et fallback première colonne)
                ts_for_df = ts_col
                if ts_for_df not in df.columns:
                    # 1) tentative insensible à la casse / espaces
                    lowered = {c.strip().lower(): c for c in df.columns}
                    key = ts_for_df.strip().lower()
                    if key in lowered:
                        real_col = lowered[key]
                        logger.info(
                            f"Colonne timestamp '{ts_for_df}' non trouvée telle quelle dans {f.name}, "
                            f"utilisation de la colonne existante '{real_col}'."
                        )
                        ts_for_df = real_col
                    else:
                        # 2) fallback simple : première colonne
                        if not df.columns:
                            raise ValueError(f"Aucune colonne lisible dans {f.name}")
                        ts_for_df = df.columns[0]
                        logger.warning(
                            f"Colonne timestamp '{ts_col}' absente dans {f.name}, "
                            f"fallback sur la première colonne '{ts_for_df}'."
                        )

                # Si CSV au format "timestamp;capteur:valeur;...", reparser en mode best-effort
                if f.suffix.lower() != ".parquet":
                    if df.shape[1] == 1:
                        sample = str(df.select(df.columns[0]).head(1).to_series()[0]) if df.height > 0 else ""
                        if ";" in sample and ":" in sample:
                            logger.info(f"Détection format raw ';capteur:valeur' pour {f.name}, parsing best-effort")
                            df = _parse_semicolon_kv_file(f)
                            report["rows_before"] = df.height
                            ts_for_df = "Time"

                df = format_timestamp_column_utc_z(df, ts_for_df)

                # Inférer device_id si absent
                if "device_id" not in df.columns:
                    device_id = get_device_id_from_stem(f.stem) or ""
                    df = df.with_columns(pl.lit(device_id).alias("device_id"))

                long_df = _raw_wide_to_long(
                    df,
                    time_col=ts_for_df,
                    device_id_col="device_id",
                    exclude_columns=exclude_columns,
                )

                if include_sensors and isinstance(include_sensors, list):
                    include_set = {str(s) for s in include_sensors}
                    long_df = long_df.filter(pl.col("sensor").is_in(include_set))

                # Si aucune donnée exploitable, on passe au fichier suivant
                if long_df.height == 0:
                    report["rows_after"] = 0
                    processed_reports.append(report)
                    # #region agent log
                    try:
                        import json
                        from pathlib import Path as _Path
                        _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                        with open(_dbg, "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H6",
                                        "location": "81_import_raw_postgres.py:run:skip_empty",
                                        "message": "skip_empty_long_df",
                                        "data": {
                                            "file": f.name,
                                            "columns": list(long_df.columns),
                                            "rows": long_df.height,
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

                # #region agent log
                try:
                    import json
                    from pathlib import Path as _Path
                    _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H1",
                                    "location": "81_import_raw_postgres.py:run:before_agg",
                                    "message": "long_df_schema_before_agg",
                                    "data": {
                                        "file": f.name,
                                        "columns": list(long_df.columns),
                                        "dtypes": {c: str(long_df.schema.get(c)) for c in long_df.columns},
                                        "rows": long_df.height,
                                    },
                                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion

                # Agrégation éventuelle 10s/60s
                if aggregation in ("10s", "60s"):
                    # Time est déjà TIMESTAMPTZ/Utf8 ISO → on passe par une colonne datetime
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

                # #region agent log
                try:
                    import json
                    from pathlib import Path as _Path
                    _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H2",
                                    "location": "81_import_raw_postgres.py:run:before_round",
                                    "message": "value_dtype_before_round",
                                    "data": {
                                        "file": f.name,
                                        "has_value_col": "value" in long_df.columns,
                                        "value_dtype": str(long_df.schema.get("value")),
                                        "rows": long_df.height,
                                    },
                                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion

                # Arrondir à 2 décimales (skip si dtype Null)
                value_dtype = long_df.schema.get("value")
                if long_df.height > 0 and "value" in long_df.columns and value_dtype != pl.Null:
                    long_df = long_df.with_columns(
                        pl.col("value").cast(pl.Float64).round(2).alias("value")
                    )
                    # #region agent log
                    try:
                        import json
                        from pathlib import Path as _Path
                        _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                        with open(_dbg, "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H4",
                                        "location": "81_import_raw_postgres.py:run:round",
                                        "message": "round_applied",
                                        "data": {
                                            "file": f.name,
                                            "value_dtype": str(value_dtype),
                                            "rows": long_df.height,
                                        },
                                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion
                else:
                    # #region agent log
                    try:
                        import json
                        from pathlib import Path as _Path
                        _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                        with open(_dbg, "a", encoding="utf-8") as _f:
                            _f.write(
                                json.dumps(
                                    {
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "H5",
                                        "location": "81_import_raw_postgres.py:run:round",
                                        "message": "round_skipped",
                                        "data": {
                                            "file": f.name,
                                            "has_value_col": "value" in long_df.columns,
                                            "value_dtype": str(value_dtype),
                                            "rows": long_df.height,
                                        },
                                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                    # #endregion

                # #region agent log
                try:
                    import json
                    from pathlib import Path as _Path
                    _dbg = _Path("/home/jbaudry/Documents/2026/PyJAMA/.cursor/debug.log")
                    with open(_dbg, "a", encoding="utf-8") as _f:
                        _f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "H3",
                                    "location": "81_import_raw_postgres.py:run:after_round",
                                    "message": "value_dtype_after_round",
                                    "data": {
                                        "file": f.name,
                                        "value_dtype": str(long_df.schema.get("value")),
                                        "rows": long_df.height,
                                    },
                                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion

                report["rows_after"] = long_df.height
                frames.append(long_df)
                processed_reports.append(report)
                total_rows_before += report["rows_before"]
                total_rows_after += report["rows_after"]
            except Exception as e:  # pragma: no cover - robustesse
                logger.error(f"Erreur lors du traitement de {f}: {e}", exc_info=True)
                report["error"] = str(e)
                failed_reports.append(report)

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

        # Mode test: limiter le nombre de lignes insérées
        limit_rows = db_cfg.get("limit_rows")
        if isinstance(limit_rows, int) and limit_rows > 0:
            combined = combined.head(limit_rows)
            logger.info(f"Mode test: limit_rows={limit_rows} -> {combined.height} lignes à insérer")

        # Étape 4 : politique de remplacement (replace / replace_all)
        ts_db_col = schema_cfg.get("timestamp_column", "Time")
        with conn.cursor() as cur:
            if policy == "replace":
                logger.info("Policy replace: suppression des lignes dans la plage concernée...")
                # Utilise from/to si fournis, sinon min/max de combined
                if from_date or to_date:
                    # Même parser que plus haut
                    def parse_for_db(date_str: str, is_end: bool = False) -> Optional[str]:
                        if not date_str or date_str == "yyyy-mm-ddTh:m:sZ":
                            return None
                        if len(date_str) == 10 and date_str.count("-") == 2:
                            date_str = f"{date_str}T23:59:59Z" if is_end else f"{date_str}T00:00:00Z"
                        return date_str

                    min_ts_str = parse_for_db(from_date, is_end=False)
                    max_ts_str = parse_for_db(to_date, is_end=True)
                else:
                    ts_min = combined["Time"].min()
                    ts_max = combined["Time"].max()
                    min_ts_str = str(ts_min) if ts_min is not None else None
                    max_ts_str = str(ts_max) if ts_max is not None else None
                if min_ts_str and max_ts_str:
                    cur.execute(
                        f"DELETE FROM {safe_table} "
                        f"WHERE {ts_db_col} >= %s AND {ts_db_col} <= %s",
                        (min_ts_str, max_ts_str),
                    )
                    deleted = cur.rowcount
                    logger.info(f"Policy replace OK: {deleted} lignes supprimées")
            elif policy == "replace_all":
                logger.info("Policy replace_all: TRUNCATE...")
                cur.execute(f"TRUNCATE TABLE {safe_table}")
                logger.info("Policy replace_all OK")
        conn.commit()

        n_rows = combined.height
        logger.info(f"Insertion en cours: {n_rows} lignes (par lots de {BATCH_INSERT_SIZE})...")
        commit_each_batch = bool(db_cfg.get("commit_each_batch", False))
        log_each_batch = bool(db_cfg.get("log_each_batch", False))
        insert_method = db_cfg.get("insert_method", "executemany")
        page_size = db_cfg.get("page_size")
        inserted = insert_rows(
            conn,
            table_name,
            combined,
            schema=schema_cfg,
            commit_each_batch=commit_each_batch,
            log_each_batch=log_each_batch,
            insert_method=insert_method,
            page_size=page_size,
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
        f"81_import_raw_postgres terminé: {len(processed_reports)} traités, "
        f"{len(failed_reports)} échecs, {total_rows_after} lignes après traitement"
    )
    return summary

