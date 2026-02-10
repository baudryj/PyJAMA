"""Tests pour le module config_v2 — adapter v2 → v1."""

import os
import sys
import pytest

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_v2 import detect_version, adapt_v2_to_v1


# ---------------------------------------------------------------------------
# detect_version
# ---------------------------------------------------------------------------

def test_detect_version_v1():
    assert detect_version({"version": "v1", "id": "test"}) == "v1"


def test_detect_version_v1_absent():
    """Si version absente, on considère v1."""
    assert detect_version({"id": "test"}) == "v1"


def test_detect_version_v2():
    assert detect_version({"version": "v2", "id": "test"}) == "v2"


# ---------------------------------------------------------------------------
# v1 passthrough
# ---------------------------------------------------------------------------

def test_v1_passthrough():
    """Un config v1 passe sans modification."""
    v1 = {
        "id": "test_v1",
        "version": "v1",
        "description": "test",
        "input": {"input_directory": "data/foo", "file_pattern": "*.csv"},
        "output": {"output_directory": "data/bar"},
    }
    result = adapt_v2_to_v1(v1)
    assert result is v1  # même objet, pas de copie


# ---------------------------------------------------------------------------
# input mapping
# ---------------------------------------------------------------------------

def test_adapt_input_files():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {
            "source": "files",
            "files": {
                "directory": "data/FOO/00_raw",
                "pattern": "*.csv",
                "recursive": True,
                "timestamp_column": "Time",
            },
        },
        "output": {"files": {"directory": "data/FOO/10_staged"}},
    }
    v1 = adapt_v2_to_v1(v2)

    assert v1["input"]["input_directory"] == "data/FOO/00_raw"
    assert v1["input"]["file_pattern"] == "*.csv"
    assert v1["input"]["search_in_subdirectory"] == "yes"
    assert v1["input"]["timestamp_column"] == "Time"
    assert v1["input"]["from"] == ""
    assert v1["input"]["to"] == ""


def test_adapt_input_recursive_false():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d", "recursive": False}},
        "output": {"files": {"directory": "o"}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["input"]["search_in_subdirectory"] == "no"


# ---------------------------------------------------------------------------
# output files mapping
# ---------------------------------------------------------------------------

def test_adapt_output_files():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {
            "files": {
                "directory": "data/FOO/10_staged",
                "suffix": "10_staged",
                "compression": "snappy",
                "partition_by": ["day"],
                "decimal_places": {"default": 3},
                "columns": ["ts", "device_id"],
                "clear": True,
            }
        },
    }
    v1 = adapt_v2_to_v1(v2)

    assert v1["output"]["output_directory"] == "data/FOO/10_staged"
    assert v1["output"]["compression"] == "snappy"
    assert v1["output"]["partition_by"] == ["day"]
    assert v1["output"]["decimal_places"] == {"default": 3}
    assert v1["output"]["output_columns"] == ["ts", "device_id"]
    assert v1["output"]["clear"] is True
    assert v1["output"]["output_file_prefix"] == ""


def test_suffix_auto_append_now_datetime():
    """Le suffix v2 reçoit automatiquement _{NOW_DATETIME} s'il ne l'a pas déjà."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o", "suffix": "10_staged"}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["output"]["output_file_suffix"] == "10_staged_{NOW_DATETIME}"


def test_suffix_already_has_now_datetime():
    """Si le suffix contient déjà {NOW_DATETIME}, ne pas le doubler."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o", "suffix": "x_{NOW_DATETIME}"}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["output"]["output_file_suffix"] == "x_{NOW_DATETIME}"


# ---------------------------------------------------------------------------
# output database mapping
# ---------------------------------------------------------------------------

def test_adapt_output_database():
    os.environ["TEST_PG_PASS"] = "secret123"
    try:
        v2 = {
            "id": "test",
            "version": "v2",
            "mode": {"type": "batch"},
            "input": {"files": {"directory": "d"}},
            "output": {
                "database": {
                    "driver": "postgres",
                    "table": "my_table",
                    "connection": {"host": "localhost", "port": 5432, "dbname": "db", "user": "u"},
                    "password_env": "TEST_PG_PASS",
                    "autocreate": True,
                    "policy": "replace",
                    "schema": {"timestamp_column": "ts"},
                    "indexes": [{"name": "idx_ts", "columns": ["ts"]}],
                }
            },
        }
        v1 = adapt_v2_to_v1(v2)
        db = v1["output"]["database"]

        assert db["table_name"] == "my_table"
        assert db["driver"] == "postgres"
        assert db["parameters"]["host"] == "localhost"
        assert db["parameters"]["password"] == "secret123"
        assert db["autocreate"] is True
        assert db["policy"] == "replace"
        assert db["schema"]["timestamp_column"] == "ts"
        assert len(db["indexes"]) == 1
    finally:
        del os.environ["TEST_PG_PASS"]


# ---------------------------------------------------------------------------
# mode mapping
# ---------------------------------------------------------------------------

def test_adapt_mode_batch():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["input"]["from"] == ""
    assert v1["input"]["to"] == ""


def test_adapt_mode_batch_fixed():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {
            "type": "batch_fixed",
            "from": "2026-01-01T00:00:00Z",
            "to": "2026-01-31T23:59:59Z",
        },
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["input"]["from"] == "2026-01-01T00:00:00Z"
    assert v1["input"]["to"] == "2026-01-31T23:59:59Z"


def test_adapt_mode_diff():
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {
            "type": "diff",
            "diff": {"strategy": "max_ts", "scope": "global"},
        },
        "input": {"files": {"directory": "d"}},
        "output": {"database": {"driver": "postgres", "table": "t", "connection": {}}},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["output"]["database"]["auto_mode"] == "max_ts"


# ---------------------------------------------------------------------------
# transform promotion
# ---------------------------------------------------------------------------

def test_adapt_transform_promotion():
    """Les clés transform sont promues à la racine v1."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
        "transform": {
            "domain_columns": {"bio_signal": ["m0"]},
            "domain_rules": {"bio_signal": {"m0": {"min": 0, "max": 800}}},
            "domain_transfo": {"bio_signal": {"m0": {"m0_log": "log"}}},
            "unit_map": {"m0": "mV"},
            "resample_by_domain": {"bio_signal": {"freq": "1s"}},
            "aggregation_level": "10s",
            "aggregation_by_domain": {"bio_signal": {"method": "median"}},
        },
    }
    v1 = adapt_v2_to_v1(v2)

    assert v1["domain_columns"] == {"bio_signal": ["m0"]}
    assert v1["domain_rules"] == {"bio_signal": {"m0": {"min": 0, "max": 800}}}
    assert v1["domain_transfo"] == {"bio_signal": {"m0": {"m0_log": "log"}}}
    assert v1["unit_map"] == {"m0": "mV"}
    assert v1["resample_by_domain"] == {"bio_signal": {"freq": "1s"}}
    assert v1["aggregation_level"] == "10s"
    assert v1["aggregation_by_domain"] == {"bio_signal": {"method": "median"}}


def test_adapt_transform_to_input():
    """dedup et filter_by_file_date vont dans input v1."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
        "transform": {"dedup": True, "filter_by_file_date": True},
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["input"]["dedup"] is True
    assert v1["input"]["filter_by_file_date"] is True


def test_adapt_transform_to_output():
    """file_name_substitute et add_quality_column vont dans output v1."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
        "transform": {
            "file_name_substitute": [{"src": "A", "target": "B"}],
            "add_quality_column": True,
        },
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["output"]["file_name_substitute"] == [{"src": "A", "target": "B"}]
    assert v1["output"]["add_quality_column"] is True


def test_adapt_transform_db_keys():
    """Les clés transform database (aggregation, insert_method, etc.) vont dans output.database."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"database": {"driver": "postgres", "table": "t", "connection": {}}},
        "transform": {
            "aggregation": "60s",
            "aggregation_method": "median",
            "insert_method": "execute_values",
            "page_size": 10000,
            "include_metrics": ["m0"],
            "value_type": "int",
        },
    }
    v1 = adapt_v2_to_v1(v2)
    db = v1["output"]["database"]
    assert db["aggregation"] == "60s"
    assert db["aggregation_method"] == "median"
    assert db["insert_method"] == "execute_values"
    assert db["page_size"] == 10000
    assert db["include_metrics"] == ["m0"]
    assert db["value_type"] == "int"


def test_adapt_transform_timestamp_columns():
    """timestamp_column_in → racine, timestamp_column_out → output."""
    v2 = {
        "id": "test",
        "version": "v2",
        "mode": {"type": "batch"},
        "input": {"files": {"directory": "d"}},
        "output": {"files": {"directory": "o"}},
        "transform": {
            "timestamp_column_in": "Time",
            "timestamp_column_out": "ts",
        },
    }
    v1 = adapt_v2_to_v1(v2)
    assert v1["timestamp_column"] == "Time"
    assert v1["output"]["timestamp_column"] == "ts"


# ---------------------------------------------------------------------------
# Round-trip : vrai config PREMANIP_GRACE
# ---------------------------------------------------------------------------

def test_roundtrip_10_parser():
    """Le 10_parser v2 produit un v1 fonctionnel."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "PREMANIP_GRACE", "10_parser.json"
    )
    with open(config_path) as f:
        v2 = json.load(f)

    v1 = adapt_v2_to_v1(v2)

    assert v1["version"] == "v1"
    assert v1["id"] == "premanip_grace_10_parser"
    assert v1["input"]["input_directory"] == "data/PREMANIP_GRACE/00_raw"
    assert v1["input"]["file_pattern"] == "*.csv"
    assert v1["input"]["dedup"] is True
    assert v1["input"]["filter_by_file_date"] is True
    assert v1["output"]["output_directory"] == "data/PREMANIP_GRACE/10_staged/default"
    assert "10_staged" in v1["output"]["output_file_suffix"]
    assert v1["output"]["file_name_substitute"][0]["src"] == "PREMANIP_GRACE"


def test_roundtrip_81_import():
    """Le 81_import v2 produit un v1 avec auto_mode et les clés database."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "PREMANIP_GRACE",
        "81_import_staged_postgres_60s.json",
    )
    with open(config_path) as f:
        v2 = json.load(f)

    os.environ["POSTGRES_PASSWORD"] = "test_pw"
    try:
        v1 = adapt_v2_to_v1(v2)
    finally:
        del os.environ["POSTGRES_PASSWORD"]

    assert v1["output"]["database"]["auto_mode"] == "max_ts"
    assert v1["output"]["database"]["table_name"] == "PREMANIP_GRACE_STAGED_60s"
    assert v1["output"]["database"]["aggregation"] == "60s"
    assert v1["output"]["database"]["aggregation_method"] == "median"
    assert v1["output"]["database"]["insert_method"] == "execute_values"
    assert v1["output"]["database"]["page_size"] == 10000
    assert v1["output"]["database"]["parameters"]["password"] == "test_pw"
    assert v1["input"]["search_in_subdirectory"] == "yes"
