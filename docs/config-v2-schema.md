# PyJAMA — Schéma de configuration v2

## Contexte

PyJAMA pré-traite des données IoT de capteurs (valvométrie, environnement, télémétrie).
Chaque fichier JSON de configuration décrit **une étape complète** de traitement : d'où viennent les données, où vont les résultats, ce qu'on leur fait, et comment on gère le temps.

Les configs doivent être **auto-descriptives** : un LLM qui lit un fichier JSON v2 doit pouvoir expliquer ce que fait l'étape sans lire le code Python.

## Structure d'une config v2

```
{
  id              identifiant unique de l'étape
  version         "v2"
  description     phrase lisible décrivant ce que fait l'étape (pour humains et LLMs)

  mode            comment on traite dans le temps
  input           d'où viennent les données
  output          où vont les résultats
  transform       ce qu'on fait aux données (spécifique à chaque étape)
  state           où écrire les logs et le suivi d'exécution
}
```

Chaque bloc a un rôle unique. Pas de mélange.

---

## mode

Décrit **quand** et **quoi** traiter.

### batch — tout le dataset

```json
"mode": { "type": "batch" }
```

Traite tous les fichiers/lignes disponibles. Pas de filtre temporel.

### batch_fixed — fenêtre déterminée

```json
"mode": {
  "type": "batch_fixed",
  "from": "2025-12-15T00:00:00Z",
  "to": "2025-12-15T23:59:59Z"
}
```

Traite uniquement les données dans la fenêtre `[from, to]`. Timestamps UTC avec suffixe Z.

Le CLI `--from` / `--to` surcharge toujours les valeurs de la config.

### diff — incrémental

```json
"mode": {
  "type": "diff",
  "diff": {
    "strategy": "max_ts",
    "scope": "global"
  }
}
```

Regarde ce qui a déjà été traité (ex : `SELECT MAX(ts)` en base, ou `max(ts)` dans les fichiers de sortie), puis complète avec les nouvelles données.

| Champ | Valeurs | Description |
|-------|---------|-------------|
| `strategy` | `max_ts` | Reprend après le dernier timestamp traité |
| `strategy` | `missing_days` | Détecte les jours manquants et les complète |
| `scope` | `global` | Un seul curseur pour tout le dataset |
| `scope` | `per_device` | Un curseur par device_id |

### realtime — flux continu

```json
"mode": { "type": "realtime" }
```

Process longue durée branché sur un flux. L'état passe à `listening` tant que le process tourne.

---

## input

Décrit **d'où** viennent les données. Un seul champ `source` indique le type, et le bloc correspondant contient les détails.

### source: files

```json
"input": {
  "source": "files",
  "files": {
    "directory": "data/PREMANIP_GRACE/00_raw",
    "pattern": "*.csv",
    "recursive": false,
    "timestamp_column": "Time"
  }
}
```

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `directory` | string | requis | Répertoire de lecture |
| `pattern` | string | `"*.parquet"` | Glob pattern des fichiers à lire |
| `recursive` | bool | `false` | Chercher dans les sous-répertoires |
| `timestamp_column` | string | `"Time"` | Nom de la colonne timestamp dans les fichiers |

### source: database

```json
"input": {
  "source": "database",
  "database": {
    "driver": "postgres",
    "table": "my_table",
    "connection": {
      "host": "localhost",
      "port": 5432,
      "dbname": "icaging",
      "user": "icaging"
    },
    "password_env": "POSTGRES_PASSWORD",
    "timestamp_column": "time"
  }
}
```

### source: stream

```json
"input": {
  "source": "stream",
  "stream": {
    "protocol": "webrtc",
    "endpoint": "wss://...",
    "timestamp_field": "ts"
  }
}
```

Réservé pour usage futur. Le protocole et l'endpoint décrivent le flux à écouter.

---

## output

Décrit **où** vont les résultats. Même logique : `target` indique le type.

### target: files

```json
"output": {
  "target": "files",
  "files": {
    "directory": "data/PREMANIP_GRACE/10_staged/default",
    "suffix": "10_staged",
    "partition_by": ["day"],
    "compression": "snappy",
    "decimal_places": { "default": 3 },
    "columns": ["ts", "device_id", "m0", "m1"],
    "clear": false
  }
}
```

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `directory` | string | requis | Répertoire d'écriture |
| `suffix` | string | `""` | Suffixe ajouté au nom de fichier (avant l'extension) |
| `partition_by` | list | `[]` | Partitionnement Hive (ex: `["day"]`) |
| `compression` | string | `"snappy"` | Compression Parquet |
| `decimal_places` | object/int | aucun | Arrondi par colonne (`{"m0": 2, "default": 3}`) ou global |
| `columns` | list | toutes | Colonnes à écrire. Supporte `"ancien->nouveau"` pour renommer |
| `clear` | bool | `false` | Vider le répertoire de sortie avant écriture |

Le timestamp de génération est ajouté automatiquement par pyjama.py au nom de fichier. Plus besoin de `{NOW_DATETIME}` dans la config.

### target: database

```json
"output": {
  "target": "database",
  "database": {
    "driver": "postgres",
    "table": "PREMANIP_GRACE_bio_signal_10s",
    "connection": {
      "host": "pil-86",
      "port": 5432,
      "dbname": "icaging",
      "user": "icaging"
    },
    "password_env": "POSTGRES_PASSWORD",
    "policy": "replace",
    "autocreate": true,
    "schema": {
      "timestamp_column": "ts",
      "device_id_column": "device_id",
      "domain_column": "domain",
      "sensor_column": "sensor",
      "value_column": "value"
    },
    "indexes": [
      {
        "name": "idx_ts",
        "columns": ["ts"],
        "method": "btree"
      }
    ]
  }
}
```

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `driver` | string | requis | Driver de base (`"postgres"`) |
| `table` | string | requis | Nom de la table cible |
| `connection` | object | requis | Paramètres de connexion (host, port, dbname, user) |
| `password_env` | string | `"POSTGRES_PASSWORD"` | Variable d'environnement contenant le mot de passe |
| `policy` | string | `"append"` | `append` (ajouter), `replace` (supprimer la fenêtre puis insérer) |
| `autocreate` | bool | `false` | Créer la table si elle n'existe pas |
| `schema` | object | requis | Mapping des colonnes cibles |
| `indexes` | list | `[]` | Index à créer |

---

## transform

Décrit **ce qu'on fait** aux données. Contenu spécifique à chaque étape.

Le bloc `transform` ne contient que des paramètres métier. Tout ce qui concerne la source, la destination ou le mode est ailleurs.

### 10_parser

```json
"transform": {
  "dedup": true,
  "filter_by_file_date": true,
  "file_name_substitute": [
    { "src": "PREMANIP_GRACE", "target": "PREMANIP-GRACE" }
  ]
}
```

### 20_split

```json
"transform": {
  "domain_columns": {
    "bio_signal": ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11"],
    "environment": ["outdoor_temp"]
  }
}
```

### 30_clean

```json
"transform": {
  "add_quality_column": true,
  "domain_rules": {
    "bio_signal": {
      "m0": { "type": "int", "min": 0, "max": 800, "max_diff": 200 }
    },
    "environment": {
      "outdoor_temp": { "type": "float", "min": -30, "max": 60, "max_diff": 10 }
    }
  }
}
```

### 40_transfo

```json
"transform": {
  "domain_transfo": {
    "bio_signal": {
      "m0": { "m0_rac_1_x": "sqrt_inv", "m0_log_b": "log" }
    }
  }
}
```

### 50_canonical

```json
"transform": {
  "timestamp_column_in": "Time",
  "domain_columns": {
    "bio_signal": ["m0", "m1", "m0_rac_1_x", "m1_rac_1_x", "m0_log_b", "m1_log_b"],
    "environment": ["outdoor_temp"]
  },
  "unit_map": {
    "outdoor_temp": "°C"
  }
}
```

### 55_resample

```json
"transform": {
  "resample_by_domain": {
    "bio_signal":  { "freq": "1s", "interpolation": "forward", "independent_by_day": true },
    "environment": { "freq": "1m", "interpolation": "forward", "independent_by_day": true },
    "telemetry":   { "freq": "5m", "interpolation": "nearest", "independent_by_day": true }
  }
}
```

### 70_aggregated

```json
"transform": {
  "aggregation_level": "10s",
  "aggregation_by_domain": {
    "bio_signal":  { "method": "median" },
    "environment": { "method": "median" }
  }
}
```

### 80_import_postgres / 81_import_staged_postgres

```json
"transform": {
  "aggregation": "60s",
  "aggregation_method": "median",
  "include_metrics": ["m0", "m1", "m2"],
  "value_type": "int",
  "insert_method": "execute_values",
  "page_size": 10000
}
```

---

## state

Décrit **où** suivre l'exécution.

```json
"state": {
  "log": "logs/PREMANIP_GRACE/10_parser.log"
}
```

Le statut est géré par pyjama.py à l'exécution. Le fichier log reçoit des entrées JSONL :

```jsonl
{"ts": "2026-02-10T14:30:00Z", "status": "running", "id": "premanip_grace_10_parser"}
{"ts": "2026-02-10T14:30:12Z", "status": "completed", "id": "premanip_grace_10_parser", "summary": {"total_files": 5, "processed": 5, "failed": 0, "rows_before": 12000, "rows_after": 11800}}
```

Statuts possibles :
- `pending` — en attente d'exécution
- `running` — en cours de traitement
- `completed` — terminé avec succès
- `error` — terminé avec erreur
- `listening` — en écoute (mode realtime)

---

## Pipeline (drawer)

Le format drawer ne change pas :

```json
{
  "drawer": "PREMANIP_GRACE",
  "items": [
    { "run": "10_parser.py", "with": "configs/PREMANIP_GRACE/10_parser.json" },
    { "run": "20_split.py",  "with": "configs/PREMANIP_GRACE/20_split.json" },
    { "run": "30_clean.py",  "with": "configs/PREMANIP_GRACE/30_clean.json" }
  ]
}
```

---

## Compatibilité v1

Si `version` est absent ou vaut `"v1"`, pyjama.py utilise le parsing v1 actuel. Aucune migration forcée.

---

## Principes de design

1. **Auto-descriptif** — Un LLM qui lit le JSON comprend l'étape sans lire le code
2. **Un bloc = un rôle** — mode (quand), input (d'où), output (où), transform (quoi), state (suivi)
3. **Déclaratif** — La config décrit le résultat voulu, pas la mécanique
4. **Pas de magie** — Pas de placeholder implicite, pas de convention cachée
5. **Simple aujourd'hui, extensible demain** — stream et database-input sont définis mais implémentés quand le besoin arrive
