"""
Adapter v2 → v1 pour les configurations PyJAMA.

Détecte la version d'une config et convertit v2 en v1
pour que les scripts existants continuent à fonctionner sans modification.
"""

import copy
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def detect_version(config: Dict[str, Any]) -> str:
    """Retourne "v1" ou "v2" selon le contenu de la config."""
    return config.get("version", "v1")


def adapt_v2_to_v1(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convertit une config v2 en config v1 compréhensible par les scripts existants.

    Si la config n'est pas v2, la retourne telle quelle.
    """
    if detect_version(config) != "v2":
        return config

    v1 = {}

    # --- Clés de base ---
    v1["id"] = config.get("id", "")
    v1["version"] = "v1"
    v1["description"] = config.get("description", "")
    if "drawer" in config:
        v1["drawer"] = config["drawer"]

    # --- mode → input.from / input.to / output.database.auto_mode ---
    mode = config.get("mode", {})
    mode_type = mode.get("type", "batch")

    # --- input ---
    v1_input = _adapt_input(config.get("input", {}), mode, mode_type)
    v1["input"] = v1_input

    # --- output ---
    v1_output = _adapt_output(config.get("output", {}))
    v1["output"] = v1_output

    # --- mode → output.database.auto_mode pour diff ---
    if mode_type == "diff":
        diff_cfg = mode.get("diff", {})
        strategy = diff_cfg.get("strategy", "max_ts")
        if "database" not in v1["output"]:
            v1["output"]["database"] = {}
        v1["output"]["database"]["auto_mode"] = strategy

    # --- mode batch_fixed → input.from / input.to ---
    if mode_type == "batch_fixed":
        v1["input"]["from"] = mode.get("from", "")
        v1["input"]["to"] = mode.get("to", "")

    # --- transform → divers emplacements v1 ---
    _adapt_transform(config.get("transform", {}), v1)

    return v1


def _adapt_input(input_cfg: Dict, mode: Dict, mode_type: str) -> Dict:
    """Convertit le bloc input v2 en bloc input v1."""
    v1_input: Dict[str, Any] = {}

    files = input_cfg.get("files", {})

    v1_input["input_directory"] = files.get("directory", "")
    v1_input["file_pattern"] = files.get("pattern", "*.parquet")
    v1_input["timestamp_column"] = files.get("timestamp_column", "Time")

    # recursive → search_in_subdirectory
    recursive = files.get("recursive", False)
    v1_input["search_in_subdirectory"] = "yes" if recursive else "no"

    # Bornes temporelles : vides par défaut (batch), surchargées par batch_fixed
    v1_input["from"] = ""
    v1_input["to"] = ""
    v1_input["except"] = ""

    return v1_input


def _adapt_output(output_cfg: Dict) -> Dict:
    """Convertit le bloc output v2 en bloc output v1."""
    v1_output: Dict[str, Any] = {}

    # --- target: files ---
    files = output_cfg.get("files", {})
    if files:
        v1_output["output_directory"] = files.get("directory", "")

        # suffix : en v2 le suffix est juste le label, pyjama ajoute _{NOW_DATETIME}
        suffix = files.get("suffix", "")
        if suffix and "{NOW_DATETIME}" not in suffix:
            suffix = f"{suffix}_{{NOW_DATETIME}}"
        v1_output["output_file_suffix"] = suffix

        v1_output["output_file_prefix"] = ""
        v1_output["compression"] = files.get("compression", "snappy")

        if "partition_by" in files:
            v1_output["partition_by"] = files["partition_by"]
        if "decimal_places" in files:
            v1_output["decimal_places"] = files["decimal_places"]
        if "clear" in files:
            v1_output["clear"] = files["clear"]
        if "columns" in files:
            v1_output["output_columns"] = files["columns"]
        if "engine" in files:
            v1_output["engine"] = files["engine"]

    # --- target: database ---
    db = output_cfg.get("database", {})
    if db:
        v1_db: Dict[str, Any] = {}
        v1_db["driver"] = db.get("driver", "postgres")
        v1_db["table_name"] = db.get("table", "")

        # connection → parameters
        conn = db.get("connection", {})
        v1_params = dict(conn)

        # password_env → parameters.password (résolution de l'env var)
        password_env = db.get("password_env", "")
        if password_env:
            v1_params["password"] = os.environ.get(password_env, "")
        v1_db["parameters"] = v1_params

        if "autocreate" in db:
            v1_db["autocreate"] = db["autocreate"]
        if "destroy" in db:
            v1_db["destroy"] = db["destroy"]
        if "policy" in db:
            v1_db["policy"] = db["policy"]
        if "schema" in db:
            v1_db["schema"] = db["schema"]
        if "indexes" in db:
            v1_db["indexes"] = db["indexes"]

        v1_output["database"] = v1_db

    return v1_output


def _adapt_transform(transform: Dict, v1: Dict) -> None:
    """Déplace les clés transform v2 vers leurs emplacements v1."""
    if not transform:
        return

    # --- Clés qui vont dans input ---
    if "dedup" in transform:
        v1["input"]["dedup"] = transform["dedup"]
    if "filter_by_file_date" in transform:
        v1["input"]["filter_by_file_date"] = transform["filter_by_file_date"]

    # --- Clés qui vont dans output ---
    if "file_name_substitute" in transform:
        v1["output"]["file_name_substitute"] = transform["file_name_substitute"]
    if "add_quality_column" in transform:
        v1["output"]["add_quality_column"] = transform["add_quality_column"]

    # --- Clés promues à la racine ---
    root_keys = [
        "domain_columns", "domain_rules", "domain_transfo", "unit_map",
        "resample_by_domain", "aggregation_level", "aggregation_by_domain",
    ]
    for key in root_keys:
        if key in transform:
            v1[key] = transform[key]

    # --- timestamp_column_in → racine timestamp_column (pour 50_canonical) ---
    if "timestamp_column_in" in transform:
        v1["timestamp_column"] = transform["timestamp_column_in"]

    # --- timestamp_column_out → output.timestamp_column (pour 50_canonical) ---
    if "timestamp_column_out" in transform:
        v1["output"]["timestamp_column"] = transform["timestamp_column_out"]

    # --- Clés qui vont dans output.database ---
    db_keys = [
        "aggregation", "aggregation_method", "include_metrics",
        "value_type", "insert_method", "page_size",
    ]
    for key in db_keys:
        if key in transform:
            if "database" not in v1["output"]:
                v1["output"]["database"] = {}
            v1["output"]["database"][key] = transform[key]

    # --- Clés database directes (log_each_batch, commit_each_batch) ---
    db_passthrough = ["log_each_batch", "commit_each_batch"]
    for key in db_passthrough:
        if key in transform:
            if "database" not in v1["output"]:
                v1["output"]["database"] = {}
            v1["output"]["database"][key] = transform[key]
