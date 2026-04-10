"""
But : ce script charge le dataset Reddit brut, renomme les colonnes,
nettoie les labels et sauvegarde un fichier CSV propre.

Usage : python clean_data.py

Remarque : les chemins des fichiers d'entrée et de sortie sont définis
en haut du script (input_file et output_file).
"""

# modules
import pandas as pd

# paramètres
input_file = "data/reddit_comments.csv"
output_file = "data/reddit_clean.csv"


def load_data(path):
    """
    But : charger un fichier CSV.

    Entrées :
        path (str) : chemin vers le fichier CSV

    Sorties :
        valeur de retour (pd.DataFrame) : le dataset chargé
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    But : nettoyer le dataset Reddit.

    Entrées :
        df (pd.DataFrame) : dataset brut

    Sorties :
        valeur de retour (pd.DataFrame) : dataset nettoyé
    """
    df = df.rename(
        columns={"Unnamed: 0": "id", "comment": "text", "sentiment": "label"}
    )

    df["label"] = df["label"].astype(str).str.strip().str.lower().str.replace(".", "")
    df = df.dropna()  # removes missing values
    df = df.drop_duplicates()  # removes duplicates

    valid_labels = ["negative", "neutral", "positive"]
    df = df[df["label"].isin(valid_labels)]

    return df


def save_data(df, path):
    """
    But : sauvegarder un dataset dans un fichier CSV.

    Entrées :
        df (pd.DataFrame) : dataset à sauvegarder
        path (str) : chemin du fichier de sortie

    Sorties :
        aucune
    """
    df.to_csv(path, index=False)


# main : programme principal qui s'exécute lorsque l'on lance le script en tant que tel
# (mais pas si il est importé par un autre script)
if __name__ == "__main__":
    try:
        # lecture, nettoyage et sauvegarde du dataset
        df = load_data(input_file)
        df_clean = clean_data(df)
        save_data(df_clean, output_file)

        # 5 premières lignes nettoyé pour vérifier que le nettoyage s'est bien passé
        print(df_clean.head())
        print(df_clean["label"].value_counts())  # nombre d'exemples par classe
        print(f"Fichier sauvegarde : {output_file}")

    # message d'erreur précis s'il y a un problème avec un fichier
    except OSError as e:
        print(f"Erreur de fichier : {e}")
    # message d'erreur précis, tous les autres problèmes
    except Exception as e:
        print(f"Erreur inattendue : {e}")
