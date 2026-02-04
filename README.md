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

## Variables d’environnement

- **`POSTGRES_PASSWORD`** — requis pour l’étape 80_import_postgres si vous exportez vers PostgreSQL.

---

## Conventions

- **Timestamps** : UTC avec suffixe `Z` (`YYYY-MM-DDTHH:MM:SSZ`).
- **Sortie pipeline** : Parquet (compression snappy). CSV via l’utilitaire `scripts/parquet_to_csv.py` si besoin.
- **Qualité** : colonne `quality_flag` (0=OK, 1=Manquant, 2=Spike, …). Voir la doc interne (ex. `.cursor/rules/data-conventions.mdc`) pour le détail.

---

## Licence

Voir [LICENSE](LICENSE) si présent. Sinon tous droits réservés.
