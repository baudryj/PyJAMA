"""
Helpers pour extraire device_id à partir des noms de fichiers.

Convention de nommage à partir de 10_staged :
<experience>_<device_id>_<date_day>_<step>_<generated_at>.parquet
Exemples :
- PREMANIP_GRACE_pil-85_2026-01-29_10_staged_2026.02.05.T.09.01.37Z.parquet
- SAMPLE_node-01_2026-01-21_10_staged_2026.02.01.T.10.00.00Z.parquet
"""

from typing import Optional


def get_device_id_from_stem(stem: str) -> Optional[str]:
    """
    Extrait le device_id (node) du nom de fichier.

    Pattern attendu (au minimum) :
    <experience>_<device_id>_<date_day>_...
    → device_id = deuxième segment séparé par "_".

    Si le pattern ne matche pas, retourne None.
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None

