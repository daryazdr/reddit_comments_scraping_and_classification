"""
But : ce script entraîne un modèle DistilBERT pour classifier
des commentaires Reddit en trois classes de sentiment.

Usage : python train_transformer.py

Remarque : les paramètres d'entraînement (chemins des fichiers, nom du modèle,
nombre d'exemples) sont définis en haut du script.
"""

# modules
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# paramètres
train_file = "data/train.csv"
test_file = "data/test.csv"
model_name = "distilbert-base-uncased"
max_train_examples = 2000
max_test_examples = 500


def load_datasets(train_path, test_path):
    """
    But : charger les fichiers d'entrainement et de test.

    Entrées :
        train_path (str) : chemin vers le fichier train.csv
        test_path (str) : chemin vers le fichier test.csv

    Sorties :
        valeur de retour (tuple[pd.DataFrame, pd.DataFrame]) : train_df, test_df
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_hf_dataset(train_df, test_df):
    """
    But : construire un DatasetDict Hugging Face a partir de deux DataFrames.
    Hugging Face n'accepte pas les DataFrames pandas directement,
    il faut les convertir dans son propre format Dataset.

    Entrées :
        train_df (pd.DataFrame) : donnees d'entrainement
        test_df (pd.DataFrame) : donnees de test

    Sorties :
        valeur de retour (DatasetDict) : dataset Hugging Face
    """
    train_df = train_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    dataset = DatasetDict(
        {
            # index remet les index à 0, 1, 2, 3...
            # car après l'échantillonnage aléatoire les index sont dans le désordre
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )
    return dataset


def preprocess_function(examples, tokenizer):
    """
    But : tokeniser un batch de textes.

    Entrées :
        examples (dict) : batch de textes
        tokenizer : tokenizer Hugging Face

    Sorties :
        valeur de retour (dict) : textes tokenises
    """
    # truncation=True signifie que si un texte dépasse 512 tokens
    # (la limite de DistilBERT), il sera coupé
    return tokenizer(examples["text"], truncation=True)


def compute_metrics_builder():
    """
    But : preparer une fonction de calcul des metriques.

    Entrées :
        aucune

    Sorties :
        valeur de retour (function) : fonction compute_metrics
    """
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        """
        But : calculer accuracy, precision, recall et F1-score.

        Entrées :
            eval_pred (tuple) : predictions du modele et labels reels

        Sorties :
            valeur de retour (dict) : dictionnaire des metriques
        """
        predictions, labels = eval_pred
        # transforme les prédictions du modèle (scores bruts) en classes concrètes
        # prend l'indice du score le plus élevé parmi les 3 classes
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)[
                "accuracy"
            ],
            "precision": precision.compute(
                predictions=predictions,
                references=labels,
                average="weighted",  # les métriques tiennent compte du déséquilibre des classes
            )["precision"],
            "recall": recall.compute(
                predictions=predictions,
                references=labels,
                average="weighted",
            )["recall"],
            "f1": f1.compute(
                predictions=predictions,
                references=labels,
                average="weighted",
            )["f1"],
        }

    return compute_metrics


# main : programme principal qui s'exécute lorsque l'on lance le script en tant que tel
# (mais pas si il est importé par un autre script)
if __name__ == "__main__":
    try:
        # chargement des donnees, entrainement et evaluation du modele
        train_df, test_df = load_datasets(train_file, test_file)

        if len(train_df) > max_train_examples:
            train_df = train_df.sample(max_train_examples, random_state=42)

        if len(test_df) > max_test_examples:
            test_df = test_df.sample(max_test_examples, random_state=42)

        print("Train :")
        print(train_df.head())
        print()

        print("Test :")
        print(test_df.head())
        print()

        dataset = build_hf_dataset(train_df, test_df)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 1) lambda est uniquement pour pouvoir passer des arguments supplémentaires,
        # parce que .map n'accepte qu'une fonction à un seul paramètre
        # 2) au lieu de traiter les exemples un par un,
        # on les traite par groupes (batches) pour la rapidité
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
        )

        # regroupe en batche savec la même longueur des examples
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        label2id = {"negative": 0, "neutral": 1, "positive": 2}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir="results",  # dossier où les checkpoints du modèle seront sauvegardés
            learning_rate=2e-5,  # le taux d'apprentissage, petite pour ne pas trop modifier les poids pré-entraînés
            per_device_train_batch_size=4,  # on traite 4 exemples à la fois
            per_device_eval_batch_size=4,
            num_train_epochs=1,  # on passe une seule fois sur toutes les données
            weight_decay=0.01,  # régularisation qui empêche les poids de devenir trop grands
            eval_strategy="epoch",  # déclenche une évaluation à la fin de chaque époque
            save_strategy="epoch",
            # recharge automatiquement le meilleur modèle vu
            # pendant l'entraînement une fois il est terminé
            load_best_model_at_end=True,
            logging_dir="logs",
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_builder(),
        )

        trainer.train()  # lance l'entraînement

        # évalue le modèle une dernière fois sur le dataset de test
        # et affiche les métriques finales
        results = trainer.evaluate()
        print("\nResultats finaux :")
        print(results)
    except OSError as e:
        print(f"Erreur de fichier ou de modele : {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
