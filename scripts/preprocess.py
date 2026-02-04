"""
Module de prétraitement des données pour PyJAMA.
Ce module charge, nettoie et transforme les données brutes issues des expériences iCaging.
Étape 1 : Nettoyage conservateur (pas de modification du temps, pas de création de nouvelles lignes).
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from fnmatch import fnmatch

import polars as pl

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_line_to_dict(line: str) -> Dict[str, any]:
    """
    Transforme une ligne CSV iCaging en dictionnaire.
    
    Format attendu: timestamp;m0:115;m1:162;m2:225;...
    
    Args:
        line: Ligne du fichier CSV
        
    Returns:
        Dictionnaire avec 'Time' et les colonnes m0, m1, m2, etc.
    """
    parts = line.strip().split(';')
    if not parts:
        return {}
    
    result = {'Time': parts[0]}  # Première partie est le timestamp
    
    # Parser les colonnes mX:value
    for part in parts[1:]:
        if ':' in part:
            col_name, value = part.split(':', 1)
            try:
                # Essayer de convertir en nombre
                result[col_name] = float(value) if '.' in value else int(value)
            except ValueError:
                result[col_name] = value
    
    return result


def load_csv_custom(path: Path) -> pl.DataFrame:
    """
    Charge un fichier CSV iCaging avec format personnalisé.
    
    Args:
        path: Chemin vers le fichier CSV
        
    Returns:
        DataFrame avec colonnes Time, m0, m1, m2, etc.
    """
    logger.info(f"Chargement du fichier: {path}")
    
    data_rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                row_dict = parse_line_to_dict(line)
                if row_dict:
                    data_rows.append(row_dict)
            except Exception as e:
                logger.warning(f"Erreur ligne {line_num} dans {path}: {e}")
                continue
    
    if not data_rows:
        raise ValueError(f"Aucune donnée valide trouvée dans {path}")
    
    df = pl.DataFrame(data_rows)
    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def apply_rules_and_filters(df: pl.DataFrame, rules: Dict) -> pl.DataFrame:
    """
    Applique les règles de filtrage et de validation sur le DataFrame.
    
    Args:
        df: DataFrame à traiter
        rules: Dictionnaire des règles (ex: {"m0": {"type": "int", "min": -10, "max": 800}})
        
    Returns:
        DataFrame avec valeurs nettoyées selon les règles
    """
    logger.info("Application des règles et filtres")
    rules = rules or {}

    # Si aucune règle n'est définie : on conserve toutes les variables sans filtrage spécifique
    if not rules:
        logger.info("rules_and_filters vide : toutes les variables sont conservées sans filtrage spécifique.")
        return df

    # Mode rules_and_filters actif : ne garder que Time + variables listées dans les règles
    columns_to_keep = ["Time"] + list(rules.keys())
    existing = [c for c in columns_to_keep if c in df.columns]
    missing = [c for c in columns_to_keep if c not in df.columns]
    if missing:
        logger.warning(f"Colonnes définies dans rules_and_filters absentes des données: {missing}")

    df = df.select(existing)
    logger.info(f"Mode rules_and_filters actif, colonnes conservées: {existing}")

    initial_rows = df.height

    # Appliquer les règles colonne par colonne
    for col_name, rule in rules.items():
        if col_name not in df.columns:
            continue

        # Conversion de type
        if "type" in rule:
            target_type = rule["type"]
            if target_type == "int":
                df = df.with_columns(
                    pl.col(col_name).cast(pl.Int64, strict=False)
                )
            elif target_type == "float":
                df = df.with_columns(
                    pl.col(col_name).cast(pl.Float64, strict=False)
                )

        # Filtrage par min/max -> valeurs hors bornes mises à null
        if "min" in rule:
            min_val = rule["min"]
            df = df.with_columns(
                pl.when(pl.col(col_name) < min_val)
                  .then(None)
                  .otherwise(pl.col(col_name))
                  .alias(col_name)
            )
        if "max" in rule:
            max_val = rule["max"]
            df = df.with_columns(
                pl.when(pl.col(col_name) > max_val)
                  .then(None)
                  .otherwise(pl.col(col_name))
                  .alias(col_name)
            )

    # Remplissage des NaN avec forward fill puis backward fill sur les colonnes numériques
    numeric_cols = [
        name for name, dtype in df.schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    ]
    cols_to_check = [c for c in numeric_cols if c != "Time"]

    for col in cols_to_check:
        before_nan = df.select(pl.col(col).is_null().sum()).to_series(0)[0]
        df = df.with_columns(
            pl.col(col).forward_fill().backward_fill()
        )
        after_nan = df.select(pl.col(col).is_null().sum()).to_series(0)[0]
        filled = before_nan - after_nan
        if filled > 0:
            logger.info(f"Colonne {col}: {filled} NaN remplis par forward/backward fill")

    # Supprimer les lignes avec NaN persistants (après forward/backward fill)
    if cols_to_check:
        rows_before_drop = df.height
        df = df.drop_nulls(subset=cols_to_check)
        rows_dropped = rows_before_drop - df.height
        if rows_dropped > 0:
            logger.info(f"{rows_dropped} lignes supprimées (NaN persistants après forward/backward fill)")

    logger.info(f"Nettoyage terminé: {initial_rows} lignes → {df.height} lignes")
    return df


def normalize_timestamps(df: pl.DataFrame, time_column: str = 'Time') -> pl.DataFrame:
    """
    Normalise les timestamps au format ISO standard (sans modifier le temps).
    
    Args:
        df: DataFrame avec colonne Time
        time_column: Nom de la colonne de temps
        
    Returns:
        DataFrame avec timestamps normalisés
    """
    if time_column not in df.columns:
        raise ValueError(f"Colonne {time_column} non trouvée")

    logger.info("Normalisation des timestamps")

    # Parser les timestamps et les reformater en ISO standard
    time_formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%S',
    ]

    def normalize_ts(ts_str: str) -> str:
        if ts_str is None:
            return ts_str

        for fmt in time_formats:
            try:
                dt = datetime.strptime(str(ts_str), fmt)
                return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except (ValueError, TypeError):
                continue

        # Si aucun format ne fonctionne, essayer parsing automatique
        try:
            dt = datetime.fromisoformat(str(ts_str).replace("Z", ""))
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            return str(ts_str)

    df = df.with_columns(
        pl.col(time_column).map_elements(normalize_ts, return_dtype=pl.Utf8)
    )

    # Vérifier les timestamps invalides (ici: valeurs vides ou nulles)
    invalid_count = df.select(pl.col(time_column).is_null().sum()).to_series(0)[0]
    if invalid_count > 0:
        logger.warning(f"{invalid_count} timestamps invalides trouvés")

    logger.info(f"Timestamps normalisés: {df.height} lignes")
    return df


def to_gaussian(df: pl.DataFrame, exclude_cols: List[str] = None) -> pl.DataFrame:
    """
    Transforme les données en distribution gaussienne avec sqrt(1/x).
    
    Args:
        df: DataFrame à transformer
        exclude_cols: Colonnes à exclure de la transformation (ex: ['Time'])
        
    Returns:
        DataFrame transformé
    """
    if exclude_cols is None:
        exclude_cols = ['Time']
    
    logger.info("Transformation gaussienne")
    
    numeric_cols = [
        name for name, dtype in df.schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    ]

    for col in numeric_cols:
        if col in exclude_cols:
            continue

        # Appliquer sqrt(1/x) uniquement pour les valeurs strictement positives
        df = df.with_columns(
            pl.when(pl.col(col) > 0)
              .then((1.0 / pl.col(col)) ** 0.5)
              .otherwise(pl.col(col))
              .alias(col)
        )
        logger.debug(f"Transformation gaussienne appliquée sur {col}")

    logger.info("Transformation gaussienne terminée")
    return df


def to_long_format(df: pl.DataFrame, time_column: str = 'Time') -> pl.DataFrame:
    """
    Convertit le DataFrame en format long (tidy data).
    
    Format: timestamp, variable, value
    
    Args:
        df: DataFrame en format large (colonnes: Time, m0, m1, m2, ...)
        time_column: Nom de la colonne de temps
        
    Returns:
        DataFrame en format long avec colonnes: timestamp, variable, value
    """
    logger.info("Conversion en format long (tidy data)")
    
    if time_column not in df.columns:
        raise ValueError(f"Colonne {time_column} non trouvée")

    # Identifier les colonnes de variables (toutes sauf Time)
    value_columns = [c for c in df.columns if c != time_column]

    if not value_columns:
        raise ValueError("Aucune colonne de valeur trouvée")

    # Utiliser melt pour convertir en format long
    df_long = df.melt(
        id_vars=[time_column],
        value_vars=value_columns,
        variable_name="variable",
        value_name="value"
    )

    # Renommer la colonne Time en timestamp
    df_long = df_long.rename({time_column: "timestamp"})

    # Supprimer les lignes avec NaN (sécurité)
    initial_rows = df_long.height
    df_long = df_long.drop_nulls(subset=["value"])
    if df_long.height < initial_rows:
        logger.info(f"{initial_rows - df_long.height} lignes avec NaN supprimées lors de la conversion")

    logger.info(f"Format long créé: {df_long.height} lignes (timestamp, variable, value)")
    return df_long


def get_file_timestamp_range(path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Lit rapidement la première et dernière ligne d'un fichier pour obtenir la plage temporelle.
    
    Args:
        path: Chemin vers le fichier CSV
        
    Returns:
        Tuple (timestamp_début, timestamp_fin) ou (None, None) si erreur
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Première ligne
            first_line = f.readline().strip()
            if not first_line:
                return None, None
            
            # Dernière ligne (lire jusqu'à la fin)
            last_line = first_line
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        
        # Parser les timestamps
        first_ts_str = first_line.split(';')[0]
        last_ts_str = last_line.split(';')[0]
        
        formats = ['%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S']
        
        first_ts = None
        last_ts = None
        
        for fmt in formats:
            try:
                first_ts = datetime.strptime(first_ts_str, fmt)
                break
            except ValueError:
                continue
        
        for fmt in formats:
            try:
                last_ts = datetime.strptime(last_ts_str, fmt)
                break
            except ValueError:
                continue
        
        return first_ts, last_ts
        
    except Exception as e:
        logger.warning(f"Erreur lecture plage temporelle de {path}: {e}")
        return None, None


def process_single_file(input_path: Path, config: Dict) -> Tuple[Path, Dict]:
    """
    Traite un seul fichier CSV avec le pipeline de nettoyage conservateur.
    
    Pipeline:
    1. Chargement des données
    2. Application des règles et filtres (min/max → NaN → forward/backward fill → suppression NaN persistants)
    3. Normalisation des timestamps (format ISO standard)
    4. Transformation gaussienne
    5. Conversion en format long (tidy data)
    6. Export CSV standard
    
    Args:
        input_path: Chemin vers le fichier d'entrée
        config: Configuration complète
        
    Returns:
        Tuple (chemin_fichier_sortie, rapport_dict)
    """
    logger.info(f"Traitement du fichier: {input_path.name}")
    
    report = {
        'input_file': str(input_path),
        'output_file': None,
        'rows_before': 0,
        'rows_after': 0,
        'error': None
    }
    
    try:
        # 1. Chargement
        df = load_csv_custom(input_path)
        report['rows_before'] = df.height
        
        # 2. Application des règles et filtres
        rules = config.get('rules_and_filters', {})
        df = apply_rules_and_filters(df, rules)
        
        # 3. Normalisation des timestamps (sans modification du temps)
        df = normalize_timestamps(df, time_column='Time')
        
        
        # 4. Transformation gaussienne
        df = to_gaussian(df, exclude_cols=['Time'])
        
        # 5. Conversion en format long (tidy data)
        df_long = to_long_format(df, time_column='Time')
        
        output_format = config['output'].get('format', 'csv')
        
        # 6. Formater les valeurs avec le nombre de décimales spécifié (CSV uniquement)
        # En Parquet on garde les types numériques.
        decimal_places = config['output'].get('decimal_places', None)
        if decimal_places is not None and output_format != 'parquet':
            fmt = "{:." + str(decimal_places) + "f}"
            df_long = df_long.with_columns(
                pl.col("value").map_elements(
                    lambda x: fmt.format(x) if x is not None else x,
                    return_dtype=pl.Utf8,
                )
            )
            logger.info(f"Valeurs formatées avec {decimal_places} décimales")
        
        # 7. Écriture du fichier de sortie
        output_dir = Path(config['output']['output_directory'])
        output_dir.mkdir(parents=True, exist_ok=True)

        input_stem = input_path.stem
        prefix = config['output'].get('output_file_prefix', '')
        suffix = config['output'].get('output_file_suffix', '')
        ext = '.parquet' if output_format == 'parquet' else '.csv'
        output_filename = f"{prefix}{input_stem}{suffix}{ext}"
        output_path = output_dir / output_filename

        if output_format == 'parquet':
            compression = config['output'].get('compression', 'snappy')
            row_group_size = config['output'].get('row_group_size', 10000)
            if config['output'].get('dictionary') or config['output'].get('use_dictionary'):
                logger.debug("dictionary/use_dictionary ignorés pour l'instant (encodage dictionnaire non branché)")
            df_long.write_parquet(output_path, compression=compression, row_group_size=row_group_size)
        else:
            df_long.write_csv(output_path, separator=',')

        report['output_file'] = str(output_path)
        report['rows_after'] = df_long.height
        logger.info(f"Fichier traité avec succès: {output_path}")
        logger.info(f"  Format: {len(df_long)} lignes en format long (timestamp, variable, value)")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {input_path}: {e}", exc_info=True)
        report['error'] = str(e)
    
    return output_path if 'output_path' in locals() else None, report


def run(config: Dict) -> Dict:
    """
    Fonction principale de prétraitement.
    
    Args:
        config: Configuration complète (déjà avec variables substituées)
        
    Returns:
        Dictionnaire de résumé avec fichiers traités, statistiques, erreurs
    """
    logger.info("Début du pipeline de prétraitement (étape 1: nettoyage conservateur)")
    
    input_config = config.get('input', {})
    output_config = config.get('output', {})
    
    # 1. Sélection des fichiers
    input_dir = Path(input_config['input_directory'])
    file_pattern = input_config.get('file_pattern', '*.csv')
    search_subdirs = input_config.get('search_in_subdirectory', 'no').lower() == 'yes'
    except_pattern = input_config.get('except', '')
    
    # Recherche des fichiers
    if search_subdirs:
        all_files = list(input_dir.rglob(file_pattern))
    else:
        all_files = list(input_dir.glob(file_pattern))
    
    logger.info(f"{len(all_files)} fichiers trouvés avec pattern {file_pattern}")
    
    # Filtrage par pattern d'exclusion
    if except_pattern:
        filtered_files = [f for f in all_files if not fnmatch(f.name, except_pattern)]
        logger.info(f"{len(filtered_files)} fichiers après exclusion de '{except_pattern}'")
    else:
        filtered_files = all_files
    
    # Filtrage par plage temporelle
    from_date = input_config.get('from', '')
    to_date = input_config.get('to', '')
    
    if from_date or to_date:
        # Parser les dates
        def parse_date(date_str: str, is_end: bool = False) -> Optional[datetime]:
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
        
        from_dt = parse_date(from_date, is_end=False)
        to_dt = parse_date(to_date, is_end=True)
        
        if from_dt or to_dt:
            time_filtered_files = []
            for f in filtered_files:
                first_ts, last_ts = get_file_timestamp_range(f)
                if first_ts is None or last_ts is None:
                    logger.warning(f"Impossible de lire plage temporelle de {f}, fichier inclus")
                    time_filtered_files.append(f)
                    continue
                
                # Vérifier si le fichier chevauche la plage demandée
                include = True
                if from_dt and last_ts < from_dt:
                    include = False
                if to_dt and first_ts > to_dt:
                    include = False
                
                if include:
                    time_filtered_files.append(f)
                else:
                    logger.info(f"Fichier {f.name} exclu (hors plage temporelle)")
            
            filtered_files = time_filtered_files
            logger.info(f"{len(filtered_files)} fichiers après filtrage temporel")
    
    # 2. Traitement de chaque fichier
    summary = {
        'total_files': len(filtered_files),
        'processed_files': [],
        'failed_files': [],
        'total_rows_before': 0,
        'total_rows_after': 0
    }
    
    for file_path in filtered_files:
        output_path, report = process_single_file(file_path, config)
        
        if report['error']:
            summary['failed_files'].append(report)
        else:
            summary['processed_files'].append(report)
            summary['total_rows_before'] += report['rows_before']
            summary['total_rows_after'] += report['rows_after']
    
    logger.info(f"Pipeline terminé: {len(summary['processed_files'])} fichiers traités, {len(summary['failed_files'])} échecs")
    
    return summary
