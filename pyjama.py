#!/usr/bin/env python3
"""
Script orchestrateur principal pour PyJAMA.
Charge une configuration JSON et exécute le pipeline de traitement.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import importlib.util
import logging

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    Charge un fichier de configuration JSON.
    
    Args:
        config_path: Chemin vers le fichier JSON
        
    Returns:
        Dictionnaire de configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration chargée depuis {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur de parsing JSON dans {config_path}: {e}")


def compute_dynamic_vars(config: Dict, cli_from: Optional[str] = None, cli_to: Optional[str] = None) -> Dict[str, str]:
    """
    Calcule les variables dynamiques (NOW_DATETIME, etc.).
    
    Args:
        config: Configuration chargée
        
    Returns:
        Dictionnaire des variables dynamiques
    """
    vars_dict: Dict[str, str] = {}

    # NOW / NOW_DATETIME en UTC (suffixe Z)
    now = datetime.utcnow()
    now_datetime_str = now.strftime('%Y.%m.%d.T.%H.%M.%SZ')
    now_iso_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    vars_dict["NOW_DATETIME"] = now_datetime_str
    vars_dict["NOW"] = now_iso_str

    # Valeurs explicites FROM / TO (issues du CLI ou de l'environnement)
    if cli_from:
        # Autoriser {NOW} dans les valeurs CLI
        vars_dict["FROM"] = cli_from.replace("{NOW}", now_iso_str)
    if cli_to:
        vars_dict["TO"] = cli_to.replace("{NOW}", now_iso_str)

    logger.info(
        f"Variables dynamiques: NOW_DATETIME={now_datetime_str}, NOW={now_iso_str}, "
        f"FROM={vars_dict.get('FROM')}, TO={vars_dict.get('TO')}"
    )

    return vars_dict


def substitute_placeholders(obj: Any, vars_dict: Dict[str, str]) -> Any:
    """
    Remplace toutes les occurrences de {VAR} dans la configuration.
    Fonction récursive pour gérer les structures imbriquées.
    
    Args:
        obj: Objet à traiter (dict, list, str, etc.)
        vars_dict: Dictionnaire des variables à substituer
        
    Returns:
        Objet avec variables substituées
    """
    if isinstance(obj, dict):
        return {k: substitute_placeholders(v, vars_dict) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_placeholders(item, vars_dict) for item in obj]
    elif isinstance(obj, str):
        result = obj
        for var_name, var_value in vars_dict.items():
            result = result.replace(f'{{{var_name}}}', var_value)
        return result
    else:
        return obj


def parse_datetime_bounds(config_input: Dict) -> tuple:
    """
    Parse les bornes temporelles from/to de la configuration.
    
    Args:
        config_input: Section 'input' de la configuration
        
    Returns:
        Tuple (from_datetime, to_datetime) ou (None, None)
    """
    from_str = config_input.get('from', '')
    to_str = config_input.get('to', '')
    
    def parse_date(date_str: str, is_end: bool = False):
        if not date_str or date_str == 'yyyy-mm-ddTh:m:sZ':
            return None
        
        # Si format date seule (yyyy-mm-dd), ajouter heure
        if len(date_str) == 10 and date_str.count('-') == 2:
            if is_end:
                date_str = f"{date_str}T23:59:59Z"
            else:
                date_str = f"{date_str}T00:00:00Z"
        
        formats = ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    from_dt = parse_date(from_str, is_end=False)
    to_dt = parse_date(to_str, is_end=True)
    
    return from_dt, to_dt


def load_script_module(script_name: str, base_path: Path) -> Any:
    """
    Charge dynamiquement un module Python depuis le répertoire scripts.
    
    Args:
        script_name: Nom du script (ex: 'preprocess.py')
        base_path: Chemin de base du projet
        
    Returns:
        Module chargé
    """
    # Résoudre le chemin du script
    if script_name.endswith('.py'):
        script_name = script_name[:-3]
    
    script_path = base_path / 'scripts' / f'{script_name}.py'
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script non trouvé: {script_path}")
    
    # Charger le module
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module {script_name}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger.info(f"Module {script_name} chargé depuis {script_path}")
    return module


def print_summary(summary: Dict):
    """
    Affiche un résumé lisible du traitement.
    
    Args:
        summary: Résumé retourné par le module de traitement
    """
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU TRAITEMENT")
    print("=" * 60)
    print(f"Fichiers trouvés: {summary.get('total_files', 0)}")
    print(f"Fichiers traités avec succès: {len(summary.get('processed_files', []))}")
    print(f"Fichiers en échec: {len(summary.get('failed_files', []))}")
    print(f"Lignes avant traitement: {summary.get('total_rows_before', 0)}")
    print(f"Lignes après traitement: {summary.get('total_rows_after', 0)}")
    
    if summary.get('processed_files'):
        print("\nFichiers traités:")
        for report in summary['processed_files']:
            out = report.get('output_file')
            out_label = Path(out).name if out else "base de données"
            print(f"  ✓ {Path(report['input_file']).name} -> {out_label}")
            print(f"    {report.get('rows_before', 0)} lignes -> {report.get('rows_after', 0)} lignes")
    
    if summary.get('failed_files'):
        print("\nFichiers en échec:")
        for report in summary['failed_files']:
            print(f"  ✗ {Path(report['input_file']).name}")
            print(f"    Erreur: {report.get('error', 'Inconnue')}")
    
    print("=" * 60 + "\n")


def run_script(
    script_name: str,
    config_path: str,
    cli_from: Optional[str] = None,
    cli_to: Optional[str] = None,
) -> int:
    """
    Exécute un script de traitement unique à partir d'un fichier de configuration.

    Args:
        script_name: Nom du script à exécuter (ex: '10_parser.py'), passé par l'appelant (drawer ou CLI).
        config_path: Chemin vers le fichier JSON de configuration.

    Returns:
        Code de retour (0 si succès, >0 en cas d'erreur ou de fichiers en échec).
    """
    try:
        # 1. Charger la configuration
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            # Si chemin relatif, chercher depuis le répertoire du script
            script_dir = Path(__file__).parent
            cfg_path = script_dir / cfg_path

        config = load_config(cfg_path)
        base_path = Path(__file__).parent

        # 2. Calculer les variables dynamiques (NOW, NOW_DATETIME, FROM, TO)
        dynamic_vars = compute_dynamic_vars(config, cli_from=cli_from, cli_to=cli_to)

        # 3. Substituer les variables dans la configuration
        config = substitute_placeholders(config, dynamic_vars)

        # 3bis. Surcharger explicitement les bornes temporelles input.from / input.to si fournies en CLI
        input_cfg = config.get("input")
        if isinstance(input_cfg, dict):
            if cli_from:
                input_cfg["from"] = dynamic_vars.get("FROM", cli_from)
            if cli_to:
                input_cfg["to"] = dynamic_vars.get("TO", cli_to)

        # 4. Script requis (passé en paramètre par drawer ou CLI)
        if not script_name:
            raise ValueError("Le nom du script est requis (ex: 10_parser.py)")

        # Vérification de l'environnement et de polars (log de debug)
        _logpath = Path(__file__).parent / "logs" / "debug.log"
        _polars_ok, _polars_err = False, None
        try:
            __import__("polars")
            _polars_ok = True
        except Exception as e:
            _polars_err = str(e)
        try:
            _logpath.parent.mkdir(parents=True, exist_ok=True)
            with open(_logpath, "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H1",
                            "location": "pyjama.py:run_script",
                            "message": "env and polars check",
                            "data": {
                                "executable": sys.executable,
                                "polars_ok": _polars_ok,
                                "polars_err": _polars_err,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            # Ne jamais faire échouer le traitement à cause du logging de debug
            pass

        # 5. Charger le module de traitement
        module = load_script_module(script_name, base_path)

        # 6. Vérifier que le module a une fonction run
        if not hasattr(module, 'run'):
            raise AttributeError(f"Le module {script_name} doit exposer une fonction 'run(config)'")

        # 7. Exécuter le pipeline
        logger.info(f"Démarrage du traitement: {config.get('id', 'N/A')}")
        logger.info(f"Description: {config.get('description', 'N/A')}")

        summary = module.run(config)

        # 8. Afficher le résumé
        print_summary(summary)

        # 9. Code de retour
        if summary.get('failed_files'):
            return 1
        return 0

    except Exception as e:
        logger.error(f"Erreur fatale lors de l'exécution de {config_path}: {e}", exc_info=True)
        print(f"\n❌ Erreur: {e}\n", file=sys.stderr)
        return 1


def main():
    """Point d'entrée CLI de PyJAMA."""
    parser = argparse.ArgumentParser(
        description='PyJAMA - Pipeline de traitement de données',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command')

    # pyjama.py run <script> --with <config>
    run_parser = subparsers.add_parser(
        'run',
        help='Exécuter un script de traitement unique'
    )
    run_parser.add_argument(
        'script',
        type=str,
        help="Nom du script à exécuter (ex: 'preprocess.py')"
    )
    run_parser.add_argument(
        '--with',
        dest='config',
        required=True,
        help='Chemin vers le fichier JSON de configuration'
    )

    args = parser.parse_args()

    if args.command == 'run':
        exit_code = run_script(args.script, args.config)
        sys.exit(exit_code)

    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
