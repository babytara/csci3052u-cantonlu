# majority class benchmark for ld, sentiment, and nli
# TF-IDF and logistic regression for NLI

# run python -m src.naive_nli_baseline from the repo root

# train_size and max_features values can be changed and tested 

import pandas as pd
import sklearn
import json
from pathlib import Path
from datetime import datetime, timezone
import platform
import csv
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_ld, load_sentiment, load_nli, stratified_sample

# Setting directory and file paths
rootDir = Path(__file__).resolve().parents[1]
resultsDir = rootDir / "results"
baselines_path = resultsDir / "baselines.csv"
runPath = resultsDir / "runs.jsonl"

# column for the baseline report
columns = ["task", "model", "split", "n", "accuracy", "macro_f1"]

# the search space for tuning, where the validation set will pick the best pair
n_range = [(1, 2), (1, 3), (1, 4)]
c_values = [0.1, 1.0, 10.0]

# settings for nli
# the number of training pairs used since the full set has alot (can test different values). (will be a stratified sample)
train_size = 100000 

# the max TF-IDF features for each vectorizer (premise and hypothesis)
max_features = 200000

# model names
naive = "majority_class"
tfidf_model = "char_tfidf_logistic_regression"

# converts to dataframe
def dataframe_convert(rows):
    return pd.DataFrame(rows, columns = ["premise", "hypothesis", "label"])

# Scores for the accuracy and macro-f1
def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division = 0)),
    }


# checks if results are already recorded, so there aren't duplicate rows when running the code multiple times
def already_logged(task, model):
    if not baselines_path.exists():
        return False
    with baselines_path.open(newline = "", encoding = "utf-8") as f:
        for r in csv.DictReader(f):
            if r["task"] == task and r["model"] == model:
                return True            
    return False


# adds the row to the baseline csv
def append_result(row):
    resultsDir.mkdir(parents=True, exist_ok=True)
    # only writes the header if the file is new or empty
    needs_header = not baselines_path.exists() or baselines_path.stat().st_size == 0
    with baselines_path.open("a", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = columns)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

# adds a row to the json run log with the task, model, and settings used
def append_run(task, model, **settings):
    row = dict(timestamp_utc = datetime.now(timezone.utc).isoformat(), task = task, model = model, seed = 42, **settings, python_version = platform.python_version(), scikit_learn_version = sklearn.__version__)
    resultsDir.mkdir(parents=True, exist_ok=True)
    with runPath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii = False) + "\n")

# features for nli
def make_features(ngram_range, max_features):
    # separate one character TF-IDF vectorizers for premise and hypothesis.
    # min_df = 2 ignores n-grams that only appear in one text
    # sublinear_tf moderates the significance of repeated words
    def vectorizer():
        return TfidfVectorizer(analyzer = "char", ngram_range = ngram_range, max_features = max_features, min_df = 2, sublinear_tf = True)
    return ColumnTransformer([("premise", vectorizer(), "premise"), ("hypothesis", vectorizer(), "hypothesis")])


# naive benchmark
def run_majority(task, y_train, y_test):
    # skip already logged rows
    if already_logged(task, naive):
        print(f"{task}: {naive} is already logged in {baselines_path.name}.")
        return
    # finds the most common label that's in the train set
    counts = Counter(y_train)
    majority_label = counts.most_common(1)[0][0]
    # predicts the most common label for each test example and gets its scores
    test_metrics = metrics(y_test, [majority_label] * len(y_test))

    # saves the results and the run
    append_result({"task": task, "model": naive, "split": "test", "n": len(y_test), **test_metrics})
    append_run(
        task,
        naive,
        train_n = len(y_train),
        test_n = len(y_test),
        majority_label = majority_label,
        train_label_counts = dict(counts),
        test_metrics = test_metrics
    )
    print(f"{task}: majority = {majority_label!r}, accuracy = {test_metrics['accuracy']:.4f}, macro-f1 = {test_metrics['macro_f1']:.4f}")


# nli model
def run_nli_tfidf(splits):
    # skip already logged rows
    if already_logged("NLI", tfidf_model):
        print(f"NLI: {tfidf_model} is already logged in {baselines_path.name}")
        return
    train_rows, val_rows, test_rows = splits
    full_train_n = len(train_rows)

    # uses a stratified random sample of the train set if there are more training pairs than the specified train_size
    if full_train_n > train_size:
        train_rows = stratified_sample(train_rows, train_size, seed = 42)

    train, val, test = dataframe_convert(train_rows), dataframe_convert(val_rows), dataframe_convert(test_rows)
    print(f"NLI pairs: train size = {len(train)} of the full {full_train_n}, val size = {len(val)}, test size = {len(test)}")

    # checks to mae sure that the NLI train and validation labels are the same
    if set(train["label"]) != set(val["label"]):
        raise ValueError("ERROR: The NLI train and validation labels are not the same.")
 
    inputs = ["premise", "hypothesis"]
    best, trials = None, []

    # Tuning the n-gram range and c value and scoring them on the validation set
    for ngram_range in n_range:
        features = make_features(ngram_range, max_features)
        X_train = features.fit_transform(train[inputs])
        X_val = features.transform(val[inputs])
        for C in c_values:
            classifier = LogisticRegression(C = C, max_iter = 2000, random_state = 42).fit(X_train, train["label"])
            val_metrics = metrics(val["label"], classifier.predict(X_val))

            # records every trial
            trials.append({
                "ngram_range": list(ngram_range), "C": C,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"]
            })
            
            print(f"NLI: ngrams = {ngram_range}, C value = {C}, validation macro-f1 = {val_metrics['macro_f1']:.4f}")

            # Set and record the settings with the best validatio macro-f1
            if best is None or val_metrics["macro_f1"] > best["val_f1"]:
                best = {
                    "val_f1": val_metrics["macro_f1"],
                    "ngram_range": ngram_range, "C": C,
                    "features": features,
                    "classifier": classifier
                }

    # scores the best model on the test set
    test_metrics = metrics(test["label"], best["classifier"].predict(best["features"].transform(test[inputs])))

    # logs the result of the test
    append_result({"task": "NLI", "model": tfidf_model, "split": "test", "n": len(test), **test_metrics})
    append_run(
        "NLI",
        tfidf_model,
        train_n = len(train),
        full_train_n = full_train_n,
        validation_n = len(val),
        test_n = len(test),
        ngram_range = list(best["ngram_range"]),
        C = best["C"],
        analyzer = "char",
        min_df = 2,
        max_features = max_features,
        sublinear_tf = True,
        max_iter = 2000,
        paired_text = "horizontally stacked premise and hypothesis vectorizers",
        selection_metric = "validation_macro_f1",
        best_validation_macro_f1 = best["val_f1"],
        test_metrics = test_metrics,
        all_validation_trials = trials,
    )

    print(f"Best NLI: ngram range = {best['ngram_range']}, C value = {best['C']}")
    print(f"NLI test scores: {test_metrics}")



def main():
    # loads the datasets
    ld, sentiment, nli = load_ld(), load_sentiment(), load_nli()

    # naive benchmark for the tasks
    print("Majority class baselines:")
    run_majority("LD", [y for _, y in ld[0]], [y for _, y in ld[2]])
    run_majority("sentiment", [y for _, y in sentiment[0]], [y for _, y in sentiment[2]])
    run_majority("NLI", [r[2] for r in nli[0]], [r[2] for r in nli[2]])

    # TF-IDF and logistic regression for NLI
    print("\nNLI TF-IDF and logistic regression baseline:")
    run_nli_tfidf(nli)

    print(f"\nResults logged in: {baselines_path}")
    print(f"Run logged in: {runPath}")
 
 
if __name__ == "__main__":
    main()