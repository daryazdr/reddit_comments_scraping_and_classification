"""
But : ce script encode les labels du dataset nettoyé et crée
les fichiers d'entraînement et de test.

Usage : python preprocessing.py

Remarque : les chemins des fichiers d'entrée et de sortie sont définis
en haut du script (input_file, train_file, test_file).
"""

# modules
import pandas as pd
from sklearn.model_selection import train_test_split

# paramètres
input_file = "data/reddit_clean.csv"
train_file = "data/train.csv"
test_file = "data/test.csv"


def load_data(path):
    """
    But : charger un fichier CSV.

    Entrées :
        path (str) : chemin du fichier

    Sorties :
        valeur de retour (pd.DataFrame) : dataset charge
    """
    return pd.read_csv(path)


def encode_labels(df):
    """
    But : convertir les labels textuels en entiers.

    Entrées :
        df (pd.DataFrame) : dataset avec labels textuels

    Sorties :
        valeur de retour (pd.DataFrame) : dataset avec labels numeriques
    """
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    df["label"] = df["label"].map(label2id)
    return df


def split_data(df, test_size=0.2):
    """
    But : separer le dataset en train et test.

    Entrées :
        df (pd.DataFrame) : dataset encode
        test_size (float) : proportion du jeu de test

    Sorties :
        valeur de retour (tuple[pd.DataFrame, pd.DataFrame]) : train_df, test_df
    """
    # random = pour que le découpage soit toujours identique à chaque exécution.
    # stratification = la proportion de chaque classe est la même dans le train et dans le test.
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    return train_df, test_df


def save_data(df, path):
    """
    But : sauvegarder un DataFrame au format CSV.

    Entrées :
        df (pd.DataFrame) : dataset a sauvegarder
        path (str) : chemin de sortie

    Sorties :
        aucune
    """
    df.to_csv(path)


# main : programme principal qui s'exécute lorsque l'on lance le script en tant que tel
# (mais pas si il est importé par un autre script)
if __name__ == "__main__":
    try:
        # chargement, encodage, separation et sauvegarde
        df = load_data(input_file)
        df = encode_labels(df)
        train_df, test_df = split_data(df)

        save_data(train_df, train_file)
        save_data(test_df, test_file)

        print(df.head())
        print(df["label"].value_counts())
        print(f"Train : {train_df.shape}")
        print(f"Test : {test_df.shape}")
    except OSError as e:
        print(f"Erreur de fichier : {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
