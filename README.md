# Reddit Gaming Comments: Sentiment Classification

Fine-tuning DistilBERT to classify r/gaming comments as **negative**, **neutral** or **positive**, then running the model on new comments scraped live from Reddit.

## What it does

1. Cleans a labeled dataset of about 21.6k Reddit comments
2. Splits it into train and test sets (80/20, stratified)
3. Fine-tunes `distilbert-base-uncased` for 3-class classification
4. Scrapes fresh comments from r/gaming and predicts their sentiment

## Results

Evaluated on 500 test examples:

| Metric | Score |
|---|---|
| Accuracy | 0.738 |
| F1 (weighted) | 0.737 |
| Precision (weighted) | 0.742 |
| Recall (weighted) | 0.738 |

These numbers come from a small CPU-only run: 2,000 training examples, 1 epoch, about 5 minutes. Training on the full dataset for more epochs (on a GPU) should give better scores.

## Dataset

[23k Reddit Gaming Comments with Sentiments](https://www.kaggle.com/datasets/sainitishmitta04/23k-reddit-gaming-comments-with-sentiments-dataset) (Kaggle). After cleaning: 21,589 comments.

| Label | Count | Share |
|---|---|---|
| positive | 9,772 | 45% |
| neutral | 7,797 | 36% |
| negative | 4,020 | 19% |

The classes are imbalanced, so the train/test split is stratified and the metrics are weighted.

## Project structure

```
├── data/                   # put reddit_comments.csv here
├── clean_data.py           # cleans the raw dataset
├── preprocessing.py        # encodes labels, splits train/test
├── train_transformer.py    # fine-tunes DistilBERT
├── scraper.py              # fetches new comments from r/gaming
├── predict.py              # scrapes comments and predicts their sentiment
└── requirements.txt
```

## How to run

Tested with Python 3.13.

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Get the data**

Download the dataset from Kaggle and save it as `data/reddit_comments.csv`.

**3. Run the scripts in order**

```bash
python clean_data.py         # -> data/reddit_clean.csv
python preprocessing.py      # -> data/train.csv and data/test.csv
python train_transformer.py  # -> trained model in results/
python predict.py            # scrapes r/gaming and prints predictions
```

The trained model isn't included in the repo because of its size, so run `train_transformer.py` before `predict.py`.

To test the scraper on its own:

```bash
python scraper.py
```

## Limitations

- Neutral comments are the hardest to classify, since they have fewer clear sentiment cues.
- The scraper uses Reddit's public JSON API without authentication, so it may stop working if Reddit changes its rules.
- The model is trained on gaming comments only and will likely perform worse on other topics.

## References

- [Hugging Face tutorial: Sequence Classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

## Author

Darya Zdrelyuk, Master 1 in Language Industries (NLP), Université Grenoble Alpes, 2025–2026