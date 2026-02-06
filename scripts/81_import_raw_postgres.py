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
        return pl.DataFrame(
            {
                time_col: df[time_col] if time_col in df.columns else [],
                device_id_col: df[device_id_col] if device_id_col in df.columns else [],
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

    with conn.cursor() as cur:
        total = 0
        for i in range(0, len(rows), BATCH_INSERT_SIZE):
            batch = rows[i : i + BATCH_INSERT_SIZE]
            cur.executemany(
                insert_sql,
                [
                    (
                        r.get("Time"),
                        str(r.get("device_id", "")),
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
    """Filtre les fichiers selon une plage from/to en lisant min/max sur la colonne temps."""

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
        # Gestion destroy/autocreate et schéma/colonnes
        with conn.cursor() as cur:
            safe_table = f'"{table_name}"' if not table_name.islower() else table_name
            if destroy:
                cur.execute(f"DROP TABLE IF EXISTS {safe_table}")
                conn.commit()
                logger.info(f"Option destroy=True : table {table_name} supprimée avant recréation")
        ensure_table_raw(conn, table_name, schema=schema_cfg)

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

                # Normaliser la colonne temps
                if ts_col not in df.columns:
                    raise ValueError(f"Colonne timestamp absente dans {f.name}: {ts_col}")
                df = format_timestamp_column_utc_z(df, ts_col)

                # Inférer device_id si absent
                if "device_id" not in df.columns:
                    device_id = get_device_id_from_stem(f.stem) or ""
                    df = df.with_columns(pl.lit(device_id).alias("device_id"))

                long_df = _raw_wide_to_long(
                    df,
                    time_col=ts_col,
                    device_id_col="device_id",
                    exclude_columns=exclude_columns,
                )

                if include_sensors and isinstance(include_sensors, list):
                    include_set = {str(s) for s in include_sensors}
                    long_df = long_df.filter(pl.col("sensor").is_in(include_set))

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

                # Arrondir à 2 décimales
                long_df = long_df.with_columns(pl.col("value").round(2).alias("value"))

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

        # Politique de remplacement : delete plage [from,to] ou TRUNCATE
        ts_db_col = schema_cfg.get("timestamp_column", "Time")
        safe_table = f'"{table_name}"' if not table_name.islower() else table_name
        with conn.cursor() as cur:
            if policy == "replace":
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
                    logger.info(
                        f"Policy replace: {deleted} lignes supprimées dans [{min_ts_str}, {max_ts_str}]"
                    )
            elif policy == "replace_all":
                cur.execute(f"TRUNCATE TABLE {safe_table}")
                logger.info(f"Policy replace_all: table {table_name} vidée")
        conn.commit()

        n_rows = combined.height
        logger.info(f"Insertion en cours: {n_rows} lignes (par lots de {BATCH_INSERT_SIZE})...")
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
    logger.info(
        f"81_import_raw_postgres terminé: {len(processed_reports)} traités, "
        f"{len(failed_reports)} échecs, {total_rows_after} lignes après traitement"
    )
    return summary

