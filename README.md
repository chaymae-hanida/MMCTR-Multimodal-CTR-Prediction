# Prédiction de CTR multimodale – WWW 2025 EReL@MIR (MicroLens-1M)

Reproduction de la baseline officielle du challenge **MM-CTR** (atelier EReL@MIR, WWW 2025) sur le jeu de données **MicroLens-1M, version x2**. Le modèle est le **DIN** (Deep Interest Network) du dépôt officiel `reczoo/WWW2025_MMCTR_Challenge`, entraîné avec la bibliothèque **FuxiCTR 2.3.10**.

## Objectif

Prédire la probabilité qu'un utilisateur clique sur une vidéo (CTR), à partir de son historique (50 positions, complété par des zéros), des informations de la vidéo cible (identifiant, étiquettes) et de son embedding multimodal de dimension 128. Métrique officielle : **AUC**.

## Données

| Ensemble | Lignes |
|---|---|
| Entraînement | 6 155 540 |
| Validation | 340 220 |
| Test (sans étiquettes) | 343 025 |

- Taux de clics de 20 % en entraînement et en validation.
- Les embeddings multimodaux sont ceux fournis par les organisateurs ; ils ne sont pas modifiés.
- Version utilisée : **x2**. Le dépôt officiel indique que la version x1 contient une fuite d'étiquettes qui gonflait l'AUC ; elle n'est pas utilisée pour les résultats ci-dessous.
- Les données ne sont pas incluses (téléchargement : ModelScope, `reczoo/MicroLens_1M_x2`).

## Méthode

- Modèle : DIN du dépôt officiel, **non modifié** (8 341 506 paramètres).
- Une seule combinaison d'hyperparamètres, tirée de la grille de la configuration officielle : dimension d'embedding 64, régularisation des embeddings 1e-6, MLP [1024, 512, 256], dropout 0,1, taux d'apprentissage 1e-3, taille de batch 8192, softmax d'attention activé.
- Entraînement : 10 époques maximum, arrêt anticipé (patience de 3), sauvegarde du meilleur modèle selon l'AUC de validation.
- Adaptations techniques : les colonnes `timestamp` et `hist_seq_ts`, inutilisées par la configuration, sont retirées des fichiers pour réduire la mémoire ; la colonne de prédiction s'appelle `Task1&2`, comme l'exige le README officiel pour la phase de recherche en cours.
- Matériel : GPU Tesla T4 (Kaggle), environ 9,5 minutes par époque.

## Résultats

| Époque | AUC (validation) | Perte d'entraînement |
|---|---|---|
| 1 | 0,7363 | 0,461 |
| 2 | 0,7519 | 0,435 |
| 3 | 0,7602 | 0,421 |
| 4 | 0,7602 | 0,406 |

- L'entraînement a été annulé pendant l'époque 5, avant le déclenchement de l'arrêt anticipé.
- Entre les époques 3 et 4, l'AUC de validation progresse de 0,00008 seulement alors que la perte d'entraînement continue de baisser : elle stagne.
- Le modèle conservé est celui de meilleure AUC de validation (0,7602), selon le mécanisme `save_best_only`.
- Fichier de prédiction : 343 025 lignes, colonnes `ID` et `Task1&2`, aucune valeur manquante, probabilité moyenne de 0,206.
- L'AUC sur le jeu de test n'est pas disponible : seules les valeurs de validation sont rapportées.

## Limites

- Le modèle n'est pas modifié : il s'agit d'une reproduction de la baseline, pas d'une amélioration.
- Une seule configuration d'hyperparamètres et une seule graine (20242025) : aucune mesure de variabilité.
- Les AUC de ce dépôt (version x2) ne sont pas comparables à celles obtenues sur la version x1.

## Contenu du dépôt

- `notebooks/01_entrainement_DIN.ipynb` : préparation des données et entraînement (le code de prédiction qu'il contient n'a pas été exécuté).
- `notebooks/02_prediction_checkpoint.ipynb` : prédiction à partir du modèle sauvegardé (il lit le modèle dans la sortie du notebook 01 sur Kaggle).
- `config/DIN_microlens_mmctr_config_long.yaml` : configuration utilisée.
- `resultats/train_long.log` : journal d'entraînement (AUC par époque).
- `resultats/DIN_MicroLens_1M_x2f_001_8c0b3740.zip` : fichier de prédiction au format de soumission.
- Le modèle sauvegardé (32 Mo) n'est pas versionné.

## Reproduction (Kaggle)

1. Créer un notebook Kaggle avec le GPU T4 et l'accès Internet activés.
2. Exécuter le notebook `01` : téléchargement du jeu x2 (environ 3 minutes), préparation, puis entraînement (jusqu'à environ 1 h 40 pour 10 époques).

## Références

- Zhu et al., *Open Benchmarking for Click-Through Rate Prediction*, CIKM 2021 (FuxiCTR).

## Auteure

Chaymae Hanida – Master MLAIM, Faculté des Sciences Dhar El Mahraz, USMBA Fès
