"""
But : ce script charge un modèle entraîné et prédit le sentiment
de commentaires récupérés automatiquement depuis Reddit.

Usage : python predict.py

Remarque : le chemin du modèle et le nombre de commentaires à récupérer
sont définis en haut du script (model_path, comments_per_post, post_limit).
"""

# modules
from transformers import pipeline
from scraper import scrape_reddit_comments

# paramètres

# chemin vers le checkpoint sauvegardé par train_transformer.py,
# numéro est choisi par Hugging Face automatiquement
model_path = "results/checkpoint-2159"
comments_per_post = 3
post_limit = 5


def load_classifier(model_path):
    """
    But : charger le modele de classification.

    Entrées :
        model_path (str) : chemin du modele

    Sorties :
        valeur de retour : pipeline de classification
    """
    # le type "text-classification" indique à Hugging Face
    # quel type de tâche c'est pour qu'il configure le pipeline correctement
    classifier = pipeline("text-classification", model=model_path)
    return classifier


def predict_comments(classifier, comments):
    """
    But : predire le sentiment d'une liste de commentaires.

    Entrées :
        classifier : modele chargé
        comments (list[str]) : commentaires à analyser

    Sorties :
        aucune
    """
    for index, comment in enumerate(comments, start=1):
        result = classifier(comment)

        print(f"Commentaire {index}/{len(comments)} :")
        print(comment)
        print("Prediction :")
        print(result)
        print()


# main : programme principal qui s'exécute lorsque l'on lance le script en tant que tel
# (mais pas si il est importé par un autre script)
if __name__ == "__main__":
    try:
        # chargement du modele, scraping des commentaires et prediction
        classifier = load_classifier(model_path)
        comments = scrape_reddit_comments(post_limit, comments_per_post)

        print(
            f"{len(comments)} commentaires recuperes "
            f"depuis {post_limit} posts Reddit.\n"
        )

        predict_comments(classifier, comments)
    except OSError as e:
        print(f"Erreur de chargement du modele : {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
