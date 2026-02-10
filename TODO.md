# TODO

## Vision : config sémantique v2

Rendre les configs JSON plus intuitives et cohérentes. Chaque étape décrit :
- **input** : d'où viennent les données (fichiers, database, flux webrtc/websocket)
- **output** : où vont les résultats (fichiers, database)
- **transform** : ce qu'on fait aux données (spécifique à chaque étape)
- **mode** : comment on traite dans le temps (voir les 4 modes ci-dessous)
- **state** : état de l'étape (pending / running / completed / error / listening) + log

Principes : descriptif > abstraction, déclaratif > magie, explicable simplement.

### Les 4 modes de traitement

1. **batch** — traitement global sur tout le dataset, sans filtre temporel
2. **batch_fixed** — traitement sur une fenêtre de temps déterminée (ex : un jour précis)
3. **batch_diff** — regarde ce qui a déjà été traité, complète avec les nouvelles données. Exécution périodique (toutes les x min/h). Doit rester cohérent avec les agrégations
4. **realtime** — branché sur un flux (fichier, webrtc) qui évolue vite. Ex : afficher un capteur sur Grafana via webrtc en quasi temps réel

### Plan d'implémentation progressive

- [x] Définir le schéma JSON v2 pour une étape générique (mode/input/output/transform/state)
- [x] Créer le module adapter `config_v2.py` (detect_version + adapt_v2_to_v1)
- [x] Intégrer la détection v2 dans `pyjama.py` (run_script)
- [x] Implémenter le suivi d'état (`state`) avec écriture dans un fichier log JSONL
- [x] Mapper les configs PREMANIP_GRACE existantes vers le schéma v2 (10 fichiers)
- [x] Écrire les tests unitaires (`tests/test_config_v2.py`, 20 tests)
- [ ] Étendre input/output pour supporter database comme source (pas seulement destination)
- [ ] Étendre input pour supporter les flux (webrtc/websocket) quand le besoin se concrétise
- [ ] Implémenter le mode `realtime` dans pyjama.py

## Décisions prises

- 2026-02-10 : la v2 est évolutive — on implémente par étapes, en commençant par `mode` et `state`
- 2026-02-10 : les configs v1 restent supportées (pas de migration forcée)
- 2026-02-10 : adapter pattern choisi — `config_v2.py` convertit v2→v1, zéro modification des scripts existants
