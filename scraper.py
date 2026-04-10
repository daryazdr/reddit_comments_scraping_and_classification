"""
But : ce script récupère des commentaires récents depuis Reddit
pour alimenter la prédiction de sentiment.

Usage : python scraper.py

Remarque : le nombre de posts et de commentaires par post sont définis
dans les paramètres par défaut de scrape_reddit_comments().
"""

# modules
import requests


def scrape_reddit_comments(post_limit=5, comments_per_post=10):
    """
    But : recuperer des commentaires Reddit sur plusieurs posts recents.

    Entrées :
        post_limit (int) : nombre maximal de posts à parcourir
        comments_per_post (int) : nombre maximal de commentaires à recuperer par post

    Sorties :
        valeur de retour (list[str]) : liste des commentaires recuperes
    """

    # reddit bloque les requêtes qui ne se présentent pas.
    # User-Agent est une étiquette qu'on envoie avec chaque requête.
    # sans ça, Reddit renvoie une erreur 403 (accès refusé).
    # la valeur "python-project" est arbitraire, n'importe quelle chaîne non vide fonctionne.
    headers = {"User-Agent": "python-project"}
    url = "https://www.reddit.com/r/gaming/.json"

    r = requests.get(url, headers=headers, timeout=10)
    # vérifie que la requête a réussi.
    # si erreur, cette ligne lève automatiquement une exception.
    r.raise_for_status()
    data = r.json()  # json --> dictionnaire python

    comments = []
    # data = tous les réponses reddit, children = liste de posts
    posts = data["data"]["children"]

    for post in posts[:post_limit]:
        post_id = post["data"]["id"]
        post_url = f"https://www.reddit.com/r/gaming/comments/{post_id}/.json"

        r_post = requests.get(post_url, headers=headers, timeout=10)
        r_post.raise_for_status()
        post_json = r_post.json()

        # vérifie que le premier est le post, le deuxième est les commentaires.
        # si la réponse n'est pas une liste ou qu'elle a moins de 2 éléments
        # (post supprimé, erreur, format inattendu), on saute ce post avec continue
        # et on passe au suivant.
        if not isinstance(post_json, list) or len(post_json) < 2:
            continue

        comment_list = post_json[1]["data"]["children"]
        collected_for_post = 0

        for comment in comment_list:
            # dans l'API Reddit, chaque élément a un champ "kind" qui indique son type,
            # "t1" = un vrai commentaire texte
            if comment["kind"] != "t1":
                continue

            body = comment["data"].get("body")
            if not body:
                continue

            comments.append(body)
            collected_for_post += 1

            if collected_for_post >= comments_per_post:
                break

    return comments


# main : programme principal qui s'exécute lorsque l'on lance le script en tant que tel
# (mais pas si il est importé par un autre script)
if __name__ == "__main__":
    try:
        # recuperation et affichage des commentaires Reddit
        comments = scrape_reddit_comments(post_limit=5, comments_per_post=10)

        print("Commentaires recuperes :\n")
        for comment in comments:
            print("-", comment)
    except Exception as e:
        print(f"Erreur pendant le scraping : {e}")
