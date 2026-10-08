# Analyse de sentiment de commentaires Reddit avec DistilBERT

Darya ZDRELYUK
Master 1 Industries de la langue, 2025-2026

## 1. Introduction

Ce projet met en place un pipeline complet d'analyse de sentiment sur des commentaires issus du subreddit r/gaming. L'objectif est d'entraîner un modèle capable de prédire automatiquement la polarité affective d'un commentaire Reddit — négatif, neutre ou positif. Il s'agit d'une tâche de classification supervisée à trois classes, réalisée avec DistilBERT fine-tuné via Hugging Face Transformers.

Le pipeline couvre toutes les étapes : nettoyage du corpus brut, encodage des labels, séparation train/test, fine-tuning du modèle, évaluation des performances (accuracy, precision, recall, F1), scraping de nouveaux commentaires Reddit en temps réel, et prédiction automatique de leur sentiment.

## 2. Lien GitHub

https://github.com/daryazdr/Reddit-comments-scraping-et-classification

## 3. Structure du projet

```
.
├── data/
│   ├── reddit_comments.csv     # Dataset brut
│   ├── reddit_clean.csv        # Généré par clean_data.py
│   ├── train.csv               # Généré par preprocessing.py
│   └── test.csv                # Généré par preprocessing.py
├── results/                    # Checkpoints du modèle (généré par train_transformer.py)
├── logs/                       # Logs d'entraînement
├── clean_data.py               # Nettoyage du corpus brut
├── preprocessing.py            # Encodage des labels + séparation train/test
├── train_transformer.py        # Fine-tuning de DistilBERT
├── predict.py                  # Prédiction sur de nouveaux commentaires Reddit
├── scraper.py                  # Scraping de commentaires depuis Reddit
├── requirements.txt            # Dépendances
└── README.md
```

## 4. Dépendances

| Bibliothèque | Usage |
|---|---|
| Python 3.13 | Langage principal |
| pandas | Chargement et manipulation des données tabulaires |
| scikit-learn | Séparation train/test stratifiée |
| transformers | DistilBERT, tokenizer, Trainer, TrainingArguments, pipeline |
| datasets | Dataset et DatasetDict Hugging Face |
| evaluate | Calcul des métriques : accuracy, precision, recall, F1 |
| torch | Backend PyTorch pour l'entraînement |
| requests | Requêtes HTTP pour le scraping Reddit |

Installation :

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install pandas scikit-learn transformers datasets evaluate torch requests
```

## 5. Exécution

### Données requises

Placer le fichier `reddit_comments.csv` dans le dossier `data/` avant de lancer quoi que ce soit. Ce fichier doit contenir trois colonnes : `Unnamed: 0` (index), `comment` (texte) et `sentiment` (label).

### Ordre d'exécution

Les scripts dépendent les uns des autres et doivent être lancés dans l'ordre :

```bash
python clean_data.py        # → data/reddit_clean.csv
python preprocessing.py     # → data/train.csv et data/test.csv
python train_transformer.py # → checkpoints dans results/
python predict.py           # scrape Reddit automatiquement + prédit le sentiment

# scraper.py peut aussi être lancé seul pour tester le scraping :
python scraper.py           # → affiche les commentaires récupérés dans le terminal
```

Remarque : le dossier results/ n'est pas inclus dans le dépôt GitHub en raison de la taille des fichiers. 
Il faut lancer train_transformer.py pour le générer avant de pouvoir utiliser predict.py.

### Contraintes matérielles

L'entraînement complet sur les 17 000 exemples n'est pas réalisable sur CPU dans des conditions raisonnables. Les paramètres par défaut limitent l'entraînement à 2 000 exemples et l'évaluation à 500 exemples, sur une seule époque (environ 5 minutes sur CPU).

## 6. Description du corpus

Le corpus est un dataset public de commentaires Reddit provenant de r/gaming, avec environ 23 000 commentaires déjà annotés en trois classes de sentiment (https://www.kaggle.com/datasets/sainitishmitta04/23k-reddit-gaming-comments-with-sentiments-dataset).

Le fichier contient les colonnes `Unnamed: 0`, `comment` et `sentiment`, renommées respectivement `id`, `text` et `label` lors du nettoyage.

Distribution des classes après nettoyage :

| Classe | Effectif |
|---|---|
| positive | 9 772 |
| neutral | 7 797 |
| negative | 4 020 |
| Total | 21 589 |

Le corpus est déséquilibré : positive représente 45 % des exemples, neutral 36 %, negative seulement 19 %. Ce déséquilibre peut influencer les performances du modèle — les classes majoritaires sont mieux apprises. La séparation train/test est donc effectuée avec stratification pour conserver ces proportions dans les deux sous-ensembles.

Après séparation (80/20) : 17 271 exemples d'entraînement et 4 318 exemples de test.

Les commentaires Reddit sont écrits dans un style conversationnel informel, parfois elliptique, avec des abréviations et des références culturelles propres à la communauté gaming, ce qui est plus difficile à traiter que du texte rédigé.

## 7. Phase d'imprégnation

La phase d'imprégnation consiste à explorer le dataset brut pour comprendre la structure des données et identifier les problèmes de qualité. Elle a été menée avec pandas : affichage des premières lignes (`head()`), vérification des colonnes, comptage des valeurs manquantes (`isnull().sum()`), visualisation de la distribution des labels (`value_counts()`).

C'est durant cette phase qu'ont été identifiés le nom peu lisible de la colonne `Unnamed: 0`, la présence potentielle de points en fin de labels (par exemple `"positive."` au lieu de `"positive"`), et le déséquilibre entre les classes. Cette étape permet d'éviter de découvrir des problèmes de données pendant l'entraînement.

## 8. Prétraitement des données

Le prétraitement est séparé en deux scripts pour distinguer le nettoyage de la préparation à l'apprentissage.

`clean_data.py` renomme les colonnes, normalise les labels (minuscules, suppression des espaces et points), supprime les doublons et les valeurs manquantes, et filtre pour ne garder que les labels valides. Le résultat est sauvegardé dans `data/reddit_clean.csv`.

`preprocessing.py` encode les labels textuels en entiers (`negative` → 0, `neutral` → 1, `positive` → 2), puis divise le corpus en train (80 %) et test (20 %) avec stratification. Les deux fichiers `data/train.csv` et `data/test.csv` sont sauvegardés avec `index=False` pour éviter la création de colonnes d'index parasites.

## 9. Modèle utilisé

Le modèle choisi est DistilBERT (`distilbert-base-uncased`). Le modèle est utilisé en mode fine-tuning : on part des poids pré-entraînés et on les ajuste sur notre tâche de classification en 3 classes.

## 10. Tokenisation et structures de données

### Avant tokenisation

Chaque exemple dans le DatasetDict Hugging Face contient :

| Champ | Type | Description |
|---|---|---|
| `text` | str | Texte brut du commentaire |
| `label` | int | Classe encodée : 0 (negative), 1 (neutral), 2 (positive) |

### Après tokenisation

`preprocess_function` applique le tokenizer de DistilBERT à chaque batch. Le tokenizer découpe les textes en sous-mots (WordPiece) et leur associe des identifiants numériques. `truncation=True` coupe les textes dépassant 512 tokens. Le padding est géré dynamiquement par `DataCollatorWithPadding`, qui aligne les séquences à la même longueur au sein de chaque batch.

Après tokenisation, chaque exemple contient :

| Champ | Type | Description |
|---|---|---|
| `input_ids` | list[int] | Identifiants des tokens, incluant `[CLS]` en début et `[SEP]` en fin |
| `attention_mask` | list[int] | 1 pour les tokens réels, 0 pour le padding |
| `label` | int | Classe cible, inchangée |

Le token `[CLS]` est celui dont la représentation finale est utilisée par la couche de classification pour produire la prédiction.

### Sorties du modèle — logits

Le modèle produit pour chaque exemple un vecteur de 3 scores bruts appelés logits :

```
[logit_negative, logit_neutral, logit_positive]
```

Ces scores ne sont pas des probabilités. La prédiction finale est obtenue par `argmax` (indice du score le plus élevé), puis convertie en label via `id2label = {0: "negative", 1: "neutral", 2: "positive"}`. Dans `predict.py`, le pipeline Hugging Face applique un softmax pour produire des probabilités et retourne le label avec son score de confiance.

## 11. Entraînement et résultats

### Paramètres

| Paramètre | Valeur |
|---|---|
| Modèle de base | distilbert-base-uncased |
| Exemples d'entraînement | 2 000 (échantillon, random_state=42) |
| Exemples de test | 500 (échantillon, random_state=42) |
| Époques | 1 |
| Learning rate | 2e-5 |
| Batch size | 4 |
| Weight decay | 0.01 |
| Évaluation | À la fin de chaque époque |

### Évolution de la loss

| Époque | Loss |
|---|---|
| 0.2 | 1.031 |
| 0.4 | 0.899 |
| 0.6 | 0.788 |
| 0.8 | 0.731 |
| 1.0 | 0.708 |

La diminution régulière confirme que le modèle apprend correctement.

### Résultats finaux (sur 500 exemples de test)

| Métrique | Score |
|---|---|
| Accuracy | 0.738 |
| Precision (weighted) | 0.742 |
| Recall (weighted) | 0.738 |
| F1-score (weighted) | 0.737 |
| Eval loss | 0.652 |

L'accuracy de 73,8 % est satisfaisante compte tenu des contraintes : CPU uniquement, 2 000 exemples, 1 époque. Un entraînement complet sur l'ensemble du corpus avec plusieurs époques améliorerait ces résultats.

## 12. Scraping et prédiction

`scraper.py` récupère des commentaires en temps réel depuis Reddit via l'API JSON publique (sans authentification). Il envoie une première requête à `r/gaming/.json` pour obtenir la liste des posts, puis une deuxième requête par post pour accéder aux commentaires. Les éléments qui ne sont pas de vrais commentaires (type `"t1"`) sont ignorés, ainsi que les commentaires vides ou supprimés.

`predict.py` charge le modèle entraîné via `pipeline("text-classification", model=model_path)`, qui encapsule tokenisation et inférence en un seul appel. Pour chaque commentaire récupéré, il affiche le label prédit et le score de confiance.

## 13. API — Description des fonctions

### clean_data.py

`load_data(path: str) -> pd.DataFrame` charge un fichier CSV et retourne un DataFrame.

`clean_data(df: pd.DataFrame) -> pd.DataFrame` renomme les colonnes, normalise les labels (minuscules, suppression des espaces et points), supprime les valeurs manquantes et les doublons, filtre les labels invalides.

`save_data(df: pd.DataFrame, path: str) -> None` sauvegarde le DataFrame en CSV avec `index=False`.

### preprocessing.py

`encode_labels(df: pd.DataFrame) -> pd.DataFrame` remplace les labels textuels par des entiers via `{"negative": 0, "neutral": 1, "positive": 2}`.

`split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]` divise le corpus avec stratification (`stratify=df["label"]`) et reproductibilité (`random_state=42`).

`save_data(df: pd.DataFrame, path: str) -> None` sauvegarde en CSV avec `index=False`.

### train_transformer.py

`load_datasets(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]` charge les deux fichiers CSV.

`build_hf_dataset(train_df, test_df) -> DatasetDict` sélectionne les colonnes `text` et `label`, remet les index à zéro (`reset_index(drop=True)`), et convertit en DatasetDict Hugging Face.

`preprocess_function(examples: dict, tokenizer) -> dict` tokenise un batch de textes avec `truncation=True`.

`compute_metrics_builder() -> function` retourne `compute_metrics(eval_pred)`, une closure qui calcule accuracy, precision, recall et F1 en mode `weighted` pour tenir compte du déséquilibre des classes.

### scraper.py

`scrape_reddit_comments(post_limit: int = 5, comments_per_post: int = 10) -> list[str]` récupère des commentaires depuis r/gaming via l'API JSON publique de Reddit.

### predict.py

`load_classifier(model_path: str)` charge un pipeline Hugging Face `text-classification` depuis un checkpoint.

`predict_comments(classifier, comments: list[str]) -> None` prédit et affiche le sentiment de chaque commentaire avec son score de confiance.

---

## 14. Gestion des exceptions

Tous les scripts principaux encapsulent leur bloc `__main__` dans un `try/except` à deux niveaux. `OSError` capture les erreurs de fichiers (fichier introuvable, permissions insuffisantes, chemin de modèle invalide). `Exception` capture tout le reste pour éviter un crash brutal et afficher un message lisible.

## 15. Complexité algorithmique

Les variables :

n = nombre d'exemples dans le corpus (21 589)
L = longueur moyenne d'un commentaire en tokens (30 pour Reddit)
E = nombre d'époques d'entraînement (1)
B = taille du batch (4)
d = dimension interne de DistilBERT (768)

### Chargement des données — O(n)

`pd.read_csv` lit le fichier ligne par ligne. Il fait une opération par exemple. C'est une complexité linéaire, O(n).

### Nettoyage des données — O(n)

Toutes les opérations de `clean_data` parcourent les données une fois (pour normaliser les labels, supprimer les doublons, supprimer les valeurs manquantes, filtrer les labels invalides). Renommer les colonnes est une exception : c'est O(1), c'est-à-dire un temps constant, parce qu'on modifie juste une étiquette en mémoire sans toucher aux données elles-mêmes. La complexité globale du nettoyage est donc O(n).

### Encodage des labels — O(n)

`df["label"].map(label2id)` remplace chaque valeur textuelle par un entier. La table de correspondance n'a que 3 entrées, donc la consulter prend un temps constant O(1). On fait ça pour chaque ligne du corpus, ce qui donne O(n) au total.

### Séparation train/test — O(n)

`train_test_split` parcourt les n exemples pour les répartir entre train et test. La stratification ajoute un simple comptage des classes, ce qui est aussi O(n). La complexité globale reste O(n).

### Construction du DatasetDict — O(n)

`Dataset.from_pandas` copie chaque exemple dans le format interne de Hugging Face. Un exemple traité = une opération. Complexité : O(n).

### Tokenisation — O(n × L)

C'est la première étape où la longueur des textes entre en jeu. Pour chaque exemple, le tokenizer découpe le texte token par token et cherche chaque morceau dans le vocabulaire. Plus un texte est long, plus ça prend de temps. Le coût par exemple est donc proportionnel à sa longueur L, et pour n exemples au total : O(n × L). En pratique, les commentaires Reddit sont courts, donc L est petite et cette étape reste rapide.

### Entraînement — O(E × n × L² × d)

C'est l'étape la plus coûteuse du pipeline. La raison principale est le mécanisme d'attention de DistilBERT : pour chaque couche du modèle, chaque token doit calculer un score vis-à-vis de tous les autres tokens de la séquence. Si une séquence a L tokens, ça produit L × L scores — d'où le L².

Le d représente la taille des vecteurs impliqués dans ces calculs (768 pour DistilBERT). On multiplie ensuite par le nombre d'exemples n et le nombre d'époques E, ce qui donne :

O(E × n × L² × d)

Dans ce projet E = 1, n = 2 000, L = 30, d = 768. L'entraînement prend environ 5 minutes sur CPU, parce que les 66 millions de paramètres de DistilBERT doivent tous être mis à jour à chaque batch.

### Inférence — O(L² × d)

Une fois le modèle entraîné, prédire le sentiment d'un commentaire ne nécessite qu'un passage en avant dans le modèle, sans mettre à jour les poids. Le coût est O(L² × d) pour un seul exemple. Puisque les textes Reddit sont courts, c'est quasi-instantané.

### Résumé

| Étape | Complexité |
|---|---|
| Chargement | O(n) |
| Nettoyage | O(n) |
| Encodage des labels | O(n) |
| Séparation train/test | O(n) |
| Construction DatasetDict | O(n) |
| Tokenisation | O(n × L) |
| Entraînement | O(E × n × L² × d) |
| Inférence (1 exemple) | O(L² × d) ≈ O(1) |

Toutes les étapes de préparation des données sont O(n) ou O(n × L) et restent rapides. C'est l'entraînement du transformer qui domine tout le reste, à cause du coût quadratique en L du mécanisme d'attention.

## 16. Limites

Classe neutre difficile à classer : Les commentaires neutres contiennent moins de marqueurs lexicaux forts, ce qui les rend plus ambigus pour le modèle.

Dépendance à l'API Reddit : Le scraper utilise l'API JSON publique sans authentification. Des changements de politique Reddit peuvent bloquer les requêtes sans modification du code.

Spécialisation domaine : Le modèle est entraîné sur des commentaires gaming. Ses performances seraient probablement moindres sur d'autres domaines.

## 17. Liens

- Tutoriel suivi : [Sequence Classification — Hugging Face](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- Modèle de base : [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)