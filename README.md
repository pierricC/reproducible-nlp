# Getting started

A repository putting the emphasis on reproducible machine learning,
using the Ibmd movie reviews dataset.

To get started, simply run:

```bash
./tasks/init.sh
```

This script does several things:
-  Install vscode extensions.
-  Install peotry if not installed.
-  Create the virtualenv environment based on the pyproject.toml and activate it.
-  Install pre-commit hooks and configure git.


# Data

The dataset consists of 25,000 IMDB movie reviews, specially selected for sentiment analysis.
The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of "Negative",
and rating >=7 have a sentiment score of "Positive."
No individual movie has more than 30 reviews.

In order to get the data,
Authentificate to kaggle API and download kaggle.json under {HOME} directory.
Once it's done, run this script to download the data under /data folder:

```bash
./tasks/download_data.sh
```


# Model Registry

In order to store and version ml models, it's common practise to use a Model Registry.
To make things simpler, we will store the model and the metadata inside a sqlite DB locally.
We can then access the UI on localhost.

Fortunately, it's very easy to configure it by running:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000
```

The database has to match the one defined in the `src/conf/config.yaml` configuration file.
