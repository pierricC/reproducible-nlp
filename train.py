"""Script that train a classifier on the movie review dataset."""

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model, metrics, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.config import config as cfg

if __name__ == "__main__":

    # Read dataset
    df = pd.read_csv(cfg.DATA_PATH)

    # Label encode the target
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "Positive" else 0)

    # We only take a fraction of the data to speed up training time.
    df = df.sample(
        frac=cfg.FRACTION_SAMPLE, random_state=cfg.SEED
    ).reset_index(drop=True)

    # Target
    y = df.sentiment.values

    # Install the nlkt punkt package on the machine if not there already.
    nltk.download("punkt")

    # Cross-validation using a simple logistic regression clf.
    df["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=cfg.NB_SPLIT)

    for fold, (t, v) in tqdm(enumerate(kf.split(X=df, y=y))):
        df.loc[v, "kfold"] = fold

    for fold in tqdm(range(cfg.NB_SPLIT)):
        train_df = df[df.kfold != fold].reset_index(drop=True)
        test_df = df[df.kfold == fold].reset_index(drop=True)

        count_vec = CountVectorizer(
            tokenizer=word_tokenize, token_pattern=None
        )

        count_vec.fit(train_df.review)

        x_train = count_vec.transform(train_df.review)
        x_test = count_vec.transform(test_df.review)

        clf = linear_model.LogisticRegression()

        clf.fit(x_train, train_df.sentiment)

        predictions = clf.predict(x_test)

        accuracy = metrics.accuracy_score(test_df.sentiment, predictions)

        print(f"Fold: {fold}, accurary: {accuracy}\n")
