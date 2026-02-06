# PyJAMA

Pipeline de traitement de données IoT (ex. valvométrie), reproductible et piloté par des configs JSON.

**Philosophie** — PyJAMA permet de traiter les données comme les ingénieurs le conçoivent : une action à la fois, bien délimitée, et toujours facile à retirer ou à rejouer. Explicite plutôt qu’implicite, fichiers plutôt que BDD pour la config, reproductibilité et lisibilité. Si un pipeline ne peut pas être compris en lisant son fichier JSON, il n’a pas sa place dans PyJAMA.

---

## Prérequis

- Python 3.x
- pip

---

## Installation

```bash
git clone <url-du-repo> PyJAMA && cd PyJAMA
./manager.sh install pyjama
```

Ou manuellement : `pip install -r requirements.txt`

---

## Structure du projet

```
configs/<EXPOSURE>/     # Configs JSON par traitement (pipeline.json, 10_parser.json, …)
data/<EXPOSURE>/       # Données par étape (00_raw → 70_aggregated)
scripts/               # Étapes du pipeline (10_parser, 20_split, … 80_import_postgres)
pyjama.py              # Exécution d’une étape
drawer.py              # Exécution d’un pipeline (orchestration)
manager.sh             # create / archive / delete / install
```

---

## Démarrage rapide

1. **Créer un traitement**  
   `./manager.sh create MON_EXPO`

2. **Lancer le pipeline**  
   `python3 drawer.py run configs/MON_EXPO/pipeline.json`

3. **Filtrage temporel** (optionnel)  
   `python3 drawer.py run configs/MON_EXPO/pipeline.json --from 2026-02-04T00:00:00Z --to 2026-02-04T23:59:59Z`  
   Format UTC avec suffixe `Z`. `{NOW}` est autorisé dans les bornes.

---

## manager.sh

| Commande | Description |
|----------|-------------|
| `./manager.sh create <EXPOSURE_NAME>` | Créer la structure d’un traitement (data + configs + llm.txt) |
| `./manager.sh archive <EXPOSURE_NAME>` | Déplacer configs + data dans `ARCHIVES/` |
| `./manager.sh delete <EXPOSURE_NAME>` | Supprimer le traitement (à faire après archivage) |
| `./manager.sh install pyjama` | Installer les dépendances (requirements.txt) |

---

## Import vers PostgreSQL / Grafana

- **Étape 80** — import des données agrégées (70_aggregated) en format long `(ts, device_id, domain, sensor, value)` :

  ```bash
  python3 pyjama.py run 80_import_postgres.py --with configs/MON_EXPO/80_postgres_bio_signal_10s.json
  ```

- **Étape 81** — import des données **brutes** (00_raw) en format long `(Time, device_id, sensor, value)` pour visualisation rapide :

  - **Agg 60s sur un jour précis** :

    ```bash
    python3 pyjama.py run 81_import_raw_postgres.py \
      --with configs/MON_EXPO/81_import_raw_postgres_60s.json \
      --from 2026-02-05 --to 2026-02-05
    ```

  - **Agg 10s sur un jour précis** :

    ```bash
    python3 pyjama.py run 81_import_raw_postgres.py \
      --with configs/MON_EXPO/81_import_raw_postgres_10s.json \
      --from 2026-02-05 --to 2026-02-05
    ```

  - **Mode auto (batch quasi live)** — le script regarde ce qui manque en base et complète à partir de `00_raw` :

    ```bash
    python3 pyjama.py run 81_import_raw_postgres.py \
      --with configs/MON_EXPO/81_import_raw_postgres_60s.json \
      --mode auto
    ```

    Même chose en 10s avec la config `81_import_raw_postgres_10s.json`.

## Variables d’environnement

- **`POSTGRES_PASSWORD`** — requis pour l’étape 80_import_postgres si vous exportez vers PostgreSQL.

---

## Conventions

- **Timestamps** : UTC avec suffixe `Z` (`YYYY-MM-DDTHH:MM:SSZ`).
- **Sortie pipeline** : Parquet (compression snappy). CSV via l’utilitaire `scripts/parquet_to_csv.py` si besoin.
- **Import Postgres (80/81)** : tables longues simples pour Grafana, colonnes clefs :
  - 80 : `ts`, `device_id`, `domain`, `sensor`, `value`
  - 81 : `Time`, `device_id`, `sensor`, `value`
- **Qualité** : colonne `quality_flag` (0=OK, 1=Manquant, 2=Spike, …). Voir la doc interne (ex. `.cursor/rules/data-conventions.mdc`) pour le détail.

---

## Licence

Voir [LICENSE](LICENSE) si présent. Sinon tous droits réservés.
