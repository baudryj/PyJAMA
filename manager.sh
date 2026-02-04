#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# manager.sh — Gestion des traitements PyJAMA
#
# Usage:
#   ./manager.sh create <EXPOSURE_NAME>   — créer la structure d'un traitement
#   ./manager.sh archive <EXPOSURE_NAME> — déplacer configs + data dans ARCHIVES/
#   ./manager.sh delete <EXPOSURE_NAME>  — supprimer le traitement (après archivage)
#   ./manager.sh install pyjama         — installer les dépendances Python
# ------------------------------------------------------------

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCHIVES_DIR="$ROOT/ARCHIVES"

usage() {
  echo "Usage: $0 { create | archive | delete | install } [ARG...]"
  echo "  create <EXPOSURE_NAME>   Créer la structure d'un traitement PyJAMA"
  echo "  archive <EXPOSURE_NAME>  Déplacer configs + data dans ARCHIVES/"
  echo "  delete <EXPOSURE_NAME>   Supprimer le traitement (à faire après archive)"
  echo "  install pyjama          Installer les dépendances (requirements.txt)"
  exit 1
}

# ---------- create ----------
cmd_create() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 create <EXPOSURE_NAME>"
    exit 1
  fi
  local EXPO="$1"
  local EXPO_DIR="$ROOT/data/$EXPO"
  local CONF_DIR="$ROOT/configs/$EXPO"
  local SCRIPTS_DIR="$ROOT/scripts"

  mkdir -p "$EXPO_DIR" "$CONF_DIR" "$SCRIPTS_DIR"

  local steps=(
    "00_raw"
    "10_staged"
    "20_split"
    "30_clean"
    "40_transfo"
    "50_canonical"
    "55_resampled"
    "60_enriched"
    "70_aggregated"
    "90_analytics_ready"
  )
  local s d f w v
  for s in "${steps[@]}"; do
    mkdir -p "$EXPO_DIR/$s"
  done

  local domains=(
    "domain=bio_signal"
    "domain=environment"
    "domain=telemetry"
  )
  local domain_steps=("20_split" "30_clean" "40_transfo" "50_canonical")
  for s in "${domain_steps[@]}"; do
    for d in "${domains[@]}"; do
      mkdir -p "$EXPO_DIR/$s/$d"
    done
  done

  declare -A domain_freq
  domain_freq["domain=bio_signal"]="freq=1s"
  domain_freq["domain=environment"]="freq=1min"
  domain_freq["domain=telemetry"]="freq=5min"
  for d in "${domains[@]}"; do
    f="${domain_freq[$d]}"
    mkdir -p "$EXPO_DIR/55_resampled/$d/$f"
    mkdir -p "$EXPO_DIR/60_enriched/$d/$f"
  done

  local windows=("window=1min" "window=1h" "window=1d")
  for w in "${windows[@]}"; do
    mkdir -p "$EXPO_DIR/70_aggregated/$w"
  done

  local views=("bio_only" "bio_plus_env" "bio_plus_env_plus_telemetry")
  for v in "${views[@]}"; do
    mkdir -p "$EXPO_DIR/90_analytics_ready/$v"
  done

  touch "$CONF_DIR/pipeline.json" "$CONF_DIR/preprocess.json" "$CONF_DIR/split.json" \
        "$CONF_DIR/clean.json" "$CONF_DIR/transfo.json" "$CONF_DIR/canonical.json" \
        "$CONF_DIR/resample.json" "$CONF_DIR/enrich.json" \
        "$CONF_DIR/agg_1min.json" "$CONF_DIR/agg_1h.json" "$CONF_DIR/agg_1d.json"

  for s in "${steps[@]}"; do
    if [[ ! -f "$EXPO_DIR/$s/.keep" ]]; then
      echo "keep" > "$EXPO_DIR/$s/.keep"
    fi
  done

  cat > "$EXPO_DIR/llm.txt" <<'LLMEOF'
LLM GUIDE — Exposure directory structure (KISS + replayable)

Context
- 1 exposure = 1 directory: data/<EXPOSURE_NAME>/
- 1 file = 1 day (created during runs): day=YYYY-MM-DD/<meaningful_filename>.parquet|jsonl
- Each step writes to its own directory. Steps never overwrite previous steps.
- Domain routing keeps signal pipelines clean while preserving joinability later.

Top-level steps (inside an exposure)
00_raw/
  - Immutable raw inputs (e.g., jsonl, csv, binary). No edits.
10_staged/
  - Parsed / readable / basic dedup. Still close to source.
20_split/
  - Routing by domain (no scientific transformation): domain=bio_signal|environment|telemetry
30_clean/
  - Cleaning rules per domain (outliers, impossible values, recording errors). Adds quality_flag (int, see table below).
40_transfo/
  - Unit conversions, calibration, mathematical transforms (log, scaling) per domain.
50_canonical/
  - Canonical long-format schema (stable columns) per domain.
55_resampled/
  - Regular time grid + missing timestamps. Optional interpolation.
  - Always marks: is_interpolated, gap_duration, quality = ok/estimated/missing.
  - Frequencies differ by domain:
    - bio_signal: freq=1s (example)
    - environment: freq=1min
    - telemetry: freq=5min
60_enriched/
  - Feature engineering (rolling stats, derivatives, event flags), per domain and frequency.
70_aggregated/
  - Aggregations by time window: window=1min, window=1h, window=1d
  - May aggregate from canonical/resampled/enriched depending on config.
90_analytics_ready/
  - Final analysis/ML datasets (explicit "views"):
    - bio_only/
    - bio_plus_env/
    - bio_plus_env_plus_telemetry/
  - These datasets are produced by joining aligned signals + context features.

Domains (why)
- domain=bio_signal: primary biological signal (valvometry)
- domain=environment: external context (temperature, pressure, light, humidity, pH, salinity)
- domain=telemetry: device health/context (battery volt, internal box temperature)

Naming conventions
- Prefer explicit filenames that describe content, resolution, and day:
  Example:
    valvometry_plus_environment__features__1min__2024-07-15.parquet
- Prefer partition folders using key=value (Hive-style) because:
  - self-documenting
  - query engines can prune partitions (DuckDB, Spark, etc.)
  - LLM can infer semantics without extra docs

Config separation
- Code lives in /scripts
- Parameters live in /configs/<EXPOSURE_NAME>/*.json
- Orchestrator runs sequences using pipeline.json

Key canonical columns (recommended)
- ts (UTC), device_id, metric, value, unit, domain, quality_flag
- Optional: source, sensor_id, tags, is_interpolated, gap_duration, resample_method

quality_flag : int
| Code | Signification                 |
| ---- | ----------------------------- |
| 0    | OK (valeur brute valide)      |
| 1    | Manquant (capteur silencieux) |
| 2    | Spike / saut brutal           |
| 3    | Hors plage physique           |
| 4    | Capteur déconnecté            |
| 5    | Valeur estimée (interpolée)   |
| 6    | Valeur corrigée manuellement  |

Important rules
1) Never modify 00_raw.
2) Never mix bio signal processing with telemetry in the same step output.
3) Interpolation is a hypothesis: always trace it with flags.
4) Reproducibility: same input + same config => same output.

LLMEOF

  echo "✅ Traitement créé:"
  echo "  - $EXPO_DIR"
  echo "  - $CONF_DIR"
  echo "  - $EXPO_DIR/llm.txt"
}

# ---------- archive ----------
cmd_archive() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 archive <EXPOSURE_NAME>"
    exit 1
  fi
  local EXPO="$1"
  local EXPO_DIR="$ROOT/data/$EXPO"
  local CONF_DIR="$ROOT/configs/$EXPO"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local dest="$ARCHIVES_DIR/${EXPO}_${stamp}"

  if [[ ! -d "$EXPO_DIR" ]] && [[ ! -d "$CONF_DIR" ]]; then
    echo "Erreur: ni data/$EXPO ni configs/$EXPO n'existent."
    exit 1
  fi

  mkdir -p "$dest"
  if [[ -d "$CONF_DIR" ]]; then
    mv "$CONF_DIR" "$dest/configs_$EXPO"
  fi
  if [[ -d "$EXPO_DIR" ]]; then
    mv "$EXPO_DIR" "$dest/data_$EXPO"
  fi
  echo "✅ Archivé (déplacé) vers: $dest"
}

# ---------- delete ----------
cmd_delete() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 delete <EXPOSURE_NAME>"
    exit 1
  fi
  local EXPO="$1"
  local EXPO_DIR="$ROOT/data/$EXPO"
  local CONF_DIR="$ROOT/configs/$EXPO"

  local found=""
  for d in "$ARCHIVES_DIR"/${EXPO}_*; do
    [[ -d "$d" ]] && found=1 && break
  done
  if [[ -z "${found:-}" ]]; then
    echo "Attention: aucune archive trouvée pour $EXPO dans $ARCHIVES_DIR."
    echo "Archiver d'abord avec: $0 archive $EXPO"
    read -r -p "Supprimer quand même ? [y/N] " r
    if [[ "${r:-}" != [yY] ]]; then
      echo "Annulé."
      exit 0
    fi
  fi

  if [[ -d "$CONF_DIR" ]]; then
    rm -rf "$CONF_DIR"
    echo "  supprimé: $CONF_DIR"
  fi
  if [[ -d "$EXPO_DIR" ]]; then
    rm -rf "$EXPO_DIR"
    echo "  supprimé: $EXPO_DIR"
  fi
  echo "✅ Traitement $EXPO supprimé."
}

# ---------- install pyjama ----------
cmd_install_pyjama() {
  local req="$ROOT/requirements.txt"
  if [[ ! -f "$req" ]]; then
    echo "Erreur: $req introuvable."
    exit 1
  fi
  pip install -r "$req"
  echo "✅ Dépendances PyJAMA installées."
}

# ---------- main ----------
case "${1:-}" in
  create)  shift; cmd_create "$@" ;;
  archive) shift; cmd_archive "$@" ;;
  delete)  shift; cmd_delete "$@" ;;
  install) shift; cmd_install_pyjama ;;  # arg "pyjama" optionnel
  *)       usage ;;
esac
