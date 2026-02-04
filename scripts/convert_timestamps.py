#!/usr/bin/env python3
"""
Script pour convertir les timestamps des fichiers CSV
du format '2026-01-23T00:00:00.748320Z' vers '2026-01-23T00:00:00Z'
(sans microsecondes mais avec Z pour UTC)
"""

import os
import sys
from datetime import datetime
from pathlib import Path


def convert_timestamp(timestamp_str):
    """
    Convertit un timestamp du format ISO vers format sans microsecondes.
    Gère plusieurs formats : avec/sans microsecondes, avec/sans 'Z'.
    
    Args:
        timestamp_str: Timestamp au format '2026-01-23T00:00:00.748320Z', 
                      '2026-01-23T06:52:52Z', ou '2026-01-23T13:28:04'
    
    Returns:
        Timestamp au format '2026-01-23T00:00:00Z'
    """
    # Liste des formats à essayer dans l'ordre
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',  # Avec microsecondes et Z
        '%Y-%m-%dT%H:%M:%SZ',      # Sans microsecondes mais avec Z
        '%Y-%m-%dT%H:%M:%S',       # Déjà converti (sans microsecondes ni Z)
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            # Reformater sans microsecondes mais avec Z pour UTC
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            continue
    
    # Si aucun format ne fonctionne
    raise ValueError(f"Format de timestamp non reconnu: {timestamp_str}")


def process_csv_file(input_path, output_path=None):
    """
    Traite un fichier CSV pour convertir tous les timestamps.
    
    Args:
        input_path: Chemin vers le fichier CSV d'entrée
        output_path: Chemin vers le fichier de sortie (si None, remplace le fichier d'entrée)
    """
    if output_path is None:
        output_path = input_path
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"Erreur: Le fichier {input_path} n'existe pas.")
        return False
    
    # Créer un fichier temporaire pour la sortie
    temp_path = output_path.with_suffix('.tmp')
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(temp_path, 'w', encoding='utf-8') as outfile:
            
            line_count = 0
            for line in infile:
                line = line.rstrip('\n\r')
                
                # Extraire le timestamp (première colonne avant le premier ';')
                if ';' in line:
                    timestamp_str, rest = line.split(';', 1)
                    
                    # Convertir le timestamp
                    new_timestamp = convert_timestamp(timestamp_str)
                    
                    # Réécrire la ligne avec le nouveau timestamp
                    new_line = f"{new_timestamp};{rest}\n"
                    outfile.write(new_line)
                else:
                    # Ligne sans séparateur, la garder telle quelle
                    outfile.write(f"{line}\n")
                
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"  Traité {line_count} lignes...")
        
        # Remplacer le fichier original par le fichier temporaire
        temp_path.replace(output_path)
        print(f"✓ Fichier {output_path} traité avec succès ({line_count} lignes)")
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement de {input_path}: {e}")
        # Supprimer le fichier temporaire en cas d'erreur
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    """Fonction principale."""
    # Définir les fichiers à traiter
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / '00_RAW'
    
    files_to_process = [
        data_dir / 'EXPO1_A_pil-97_2026-01-22.csv',
        data_dir / 'EXPO1_A_pil-97_2026-01-23.csv',
    ]
    
    print("Conversion des timestamps dans les fichiers CSV...")
    print("=" * 60)
    
    success_count = 0
    for file_path in files_to_process:
        print(f"\nTraitement de: {file_path.name}")
        if process_csv_file(file_path):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Conversion terminée: {success_count}/{len(files_to_process)} fichiers traités avec succès")


if __name__ == '__main__':
    main()
