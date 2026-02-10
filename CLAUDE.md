# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyJAMA is a reproducible IoT data processing pipeline framework for biological signal data (valvometry), environmental context, and device telemetry. Configuration is JSON-driven — if a pipeline can't be understood by reading its JSON config, it doesn't belong in PyJAMA.

**Dependencies:** polars (≥0.20.0), psycopg2-binary (≥2.9.0). Install with `./manager.sh install pyjama` or `pip install -r requirements.txt`.

## Commands

```bash
# Run a single processing step
python3 pyjama.py run <script> --with <config.json> [--from DATE] [--to DATE] [--mode auto]

# Run a full pipeline (sequential steps, stops on first failure)
python3 drawer.py run <pipeline.json> [--from DATE] [--to DATE] [--mode auto]

# Project lifecycle
./manager.sh create <EXPOSURE>    # scaffold directory structure
./manager.sh archive <EXPOSURE>   # backup to ARCHIVES/
./manager.sh delete <EXPOSURE>    # remove (after archiving)
```

There are no automated tests. The `tests/` directory is empty. Manual testing is done by running pipelines against sample configs in `configs/sample/`.

## Architecture

### Pipeline Stages (sequential, domain-aware)

Each exposure processes data through numbered stages in `data/<EXPOSURE>/`:

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 00 | `00_raw/` | Immutable raw inputs (CSV, JSON, binary) — never modified |
| 10 | `10_staged/` | Parsed, deduplicated (→ Parquet) |
| 20 | `20_split/` | Domain routing: `bio_signal`, `environment`, `telemetry` |
| 30 | `30_clean/` | Outlier detection, `quality_flag` added |
| 40 | `40_transfo/` | Unit conversions, calibration |
| 50 | `50_canonical/` | Standardized long format: `(ts, device_id, metric, value, unit, domain, quality_flag)` |
| 55 | `55_resampled/` | Regular time grids, interpolation |
| 60 | `60_enriched/` | Feature engineering (rolling stats, derivatives) |
| 70 | `70_aggregated/` | Time-windowed aggregations (wide Parquet) |
| 80/81 | PostgreSQL | Import to Postgres for Grafana visualization |
| 90 | `90_analytics_ready/` | Final ML-ready datasets |

### Execution Model

- **pyjama.py** — loads a JSON config, resolves placeholders (`{NOW}`, `{NOW_DATETIME}`, `{FROM}`, `{TO}`), dynamically imports the script module from `scripts/`, calls its `run(config)` function.
- **drawer.py** — reads a pipeline JSON (`drawer` name + `items` array of `{run, with}` pairs), delegates each step to `pyjama.run_script()` sequentially.
- `--mode auto` sets `config["output"]["database"]["auto_mode"] = "max_ts"` for database import steps (80/81).

### Script Contract

Every script in `scripts/` must expose:

```python
def run(config: dict) -> dict:
    # Returns:
    # {
    #   "total_files": int,
    #   "processed_files": [{"input_file", "output_file", "rows_before", "rows_after"}],
    #   "failed_files": [{"input_file", "error"}],
    #   "total_rows_before": int,
    #   "total_rows_after": int,
    # }
```

Exit code is 0 if `failed_files` is empty, 1 otherwise.

### Helper Modules (in `scripts/`)

- **format_ts.py** — `format_timestamp_column_utc_z(df, col_name)`: normalize timestamps to `YYYY-MM-DDTHH:MM:SSZ`. Must be called before every `write_parquet`.
- **output_columns_helper.py** — `apply_output_columns()`: filter/rename columns before Parquet write. Supports `old->new` syntax.
- **device_id_helper.py** — `get_device_id_from_stem()`: extract device ID from filename (2nd `_`-separated segment).

## Data Conventions

- **Timestamps**: Always UTC with `Z` suffix (`YYYY-MM-DDTHH:MM:SSZ`). Column name is `Time` for stages 10–40, `ts` for stages 50+. Configurable via `input.timestamp_column` / `output.timestamp_column`.
- **Output format**: All pipeline stages write **Parquet only** (snappy compression). Use `scripts/parquet_to_csv.py` for CSV conversion.
- **Input format**: Auto-detected by extension — `.parquet` → `pl.read_parquet`, otherwise `pl.read_csv` with `null_values="NaN"`.
- **quality_flag codes**: 0=OK, 1=Missing, 2=Spike, 3=OutOfRange, 4=Disconnected, 5=Interpolated, 6=Manually corrected.
- **File naming**: `(prefix_)base_stem(_suffix).parquet`. Suffix uses `{NOW_DATETIME}` placeholder → `YYYY.MM.DD.T.HH.MM.SSZ`.
- **decimal_places**: Optional per-column rounding config in `output.decimal_places` (`{"m0": 2, "default": 3}` or legacy integer).
- **output_columns**: Optional column filter/rename list in `output.output_columns`. Absent columns silently ignored.

## Config Structure

All configs share a common shape:

```json
{
  "id": "unique_id",
  "version": "v1",
  "description": "...",
  "input": {
    "input_directory": "data/<EXPOSURE>/<stage>",
    "timestamp_column": "Time",
    "file_pattern": "*.parquet",
    "from": "...", "to": "..."
  },
  "output": {
    "output_directory": "data/<EXPOSURE>/<next_stage>",
    "output_file_suffix": "_suffix_{NOW_DATETIME}",
    "compression": "snappy",
    "partition_by": ["day"]
  }
}
```

Database configs (80/81) add `output.database` with `driver`, `table_name`, `parameters` (host/port/dbname/user), `password_env` (env var name, typically `POSTGRES_PASSWORD`), `autocreate`, `schema`, `indexes`.

## Key Patterns

- **Immutability**: Never modify `00_raw/` or outputs of previous stages.
- **Domain separation**: Data is split at stage 20 into `bio_signal/`, `environment/`, `telemetry/` subdirectories.
- **Canonical long format**: From stage 50 onward, data follows `(ts, device_id, metric, value, unit, domain, quality_flag)`.
- **Partitioning**: Hive-style directory partitioning by `day`.
- **polars throughout**: All DataFrame operations use polars, not pandas.
