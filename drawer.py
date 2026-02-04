#!/usr/bin/env python3
"""
Orchestrateur de pipelines pour PyJAMA.

Exemples d'utilisation :

    python3 drawer.py run pipelines/data1.json

Format attendu du fichier de pipeline JSON :

{
  "drawer": "data1",
  "items": [
    { "run": "preprocess.py", "with": "confs/data1/preprocess.json" },
    { "run": "normalisation.py", "with": "confs/data1/normalized.json" },
    { "run": "aggregation.py", "with": "confs/data1/agg_1h.json" }
  ]
}

Les chemins dans le champ \"with\" sont résolus par rapport à la racine du projet PyJAMA.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pyjama


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_pipeline(path: Path) -> Dict[str, Any]:
    """Charge et valide un fichier de pipeline JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier de pipeline non trouvé: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            pipeline = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de parsing JSON dans {path}: {e}")

    if "drawer" not in pipeline or not isinstance(pipeline["drawer"], str):
        raise ValueError("Le fichier de pipeline doit contenir une clé 'drawer' (string).")
    if "items" not in pipeline or not isinstance(pipeline["items"], list):
        raise ValueError("Le fichier de pipeline doit contenir une liste 'items'.")

    for idx, item in enumerate(pipeline["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{idx} de 'items' doit être un objet.")
        if "run" not in item or "with" not in item:
            raise ValueError(f"Item #{idx} doit contenir les clés 'run' et 'with'.")

    return pipeline


def execute_pipeline(pipeline_path: str, cli_from: str | None = None, cli_to: str | None = None) -> int:
    """
    Exécute séquentiellement les étapes décrites dans un pipeline JSON.

    Args:
        pipeline_path: Chemin vers le fichier de pipeline JSON.

    Returns:
        Code de retour global (0 si toutes les étapes ont réussi).
    """
    path = Path(pipeline_path)
    if not path.is_absolute():
        # chemins de pipeline relatifs à la racine du projet (répertoire de ce fichier)
        base = Path(__file__).parent
        path = base / path

    pipeline = load_pipeline(path)
    drawer_name = pipeline.get("drawer", path.stem)

    logger.info(f"Démarrage du pipeline drawer='{drawer_name}' depuis {path}")

    project_root = Path(__file__).parent

    for idx, item in enumerate(pipeline["items"], start=1):
        script = item["run"]
        config_rel = item["with"]

        config_path = Path(config_rel)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        logger.info(
            f"[{drawer_name}] Étape {idx}/{len(pipeline['items'])} : "
            f"script='{script}', config='{config_path}'"
        )

        exit_code = pyjama.run_script(script, str(config_path), cli_from=cli_from, cli_to=cli_to)
        if exit_code != 0:
            logger.error(
                f"[{drawer_name}] Échec à l'étape {idx} avec le script '{script}' "
                f"et la config '{config_path}' (code={exit_code}). Arrêt du pipeline."
            )
            return exit_code

    logger.info(f"Pipeline drawer='{drawer_name}' terminé avec succès.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drawer - Orchestrateur de pipelines PyJAMA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Exécuter un pipeline décrit dans un fichier JSON",
    )
    run_parser.add_argument(
        "pipeline",
        type=str,
        help="Chemin vers le fichier de pipeline JSON (relatif à la racine du projet ou absolu)",
    )
    run_parser.add_argument(
        "--from",
        dest="time_from",
        type=str,
        help="Borne temporelle de début (format UTC Z, ex. 2026-02-04T10:00:00Z, peut contenir {NOW})",
    )
    run_parser.add_argument(
        "--to",
        dest="time_to",
        type=str,
        help="Borne temporelle de fin (format UTC Z, ex. 2026-02-04T11:00:00Z, peut contenir {NOW})",
    )

    args = parser.parse_args()

    if args.command == "run":
        code = execute_pipeline(args.pipeline, cli_from=args.time_from, cli_to=args.time_to)
        sys.exit(code)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()

