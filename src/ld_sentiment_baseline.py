import json
import random
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.data import load_ld, load_sentiment

SEED = 42
random.seed(SEED)

rootDir = Path(__file__).resolve().parents[1]
resultsDir = rootDir / "results"
baselines_path = resultsDir / "baselines.csv"
runPath = resultsDir / "runs.jsonl"

nRange = [(2, 5), (3, 5), (2, 6)]
cValues = [0.1, 1.0, 10.0]
modelName = "char_tfidf_logistic_regression"
columns = ["task", "model", "split", "n", "accuracy", "macro_f1"]

#Prepping the data
def unpack_split(split, task, split_name):
    if not split:
        raise ValueError(f"{task} {split_name} split is empty.")

    texts, labels = [], []
    for i, item in enumerate(split):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                f"{task} {split_name} item {i} must be (sentence, label); "
                f"got {type(item).__name__}: {item!r}"
            )
        text, label = item
        if text is None or label is None:
            continue
        text = str(text)
        if not text.strip():
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError(f"{task} {split_name} has no valid examples.")
    return texts, labels

def make_model(ngram_range, c_value):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=True,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            C=c_value,
            max_iter=2000,
            random_state=SEED,
        )),
    ])

#Calc accuracy and F1
def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
    }

#These two save the work
def append_result(row):
    resultsDir.mkdir(parents=True, exist_ok=True)
    needs_header = not baselines_path.exists() or baselines_path.stat().st_size == 0
    with baselines_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

def append_run(record):
    resultsDir.mkdir(parents=True, exist_ok=True)
    with runPath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

#Main task
def run_task(task, splits):
    train, val, test = splits
    X_train, y_train = unpack_split(train, task, "train")
    X_val, y_val = unpack_split(val, task, "validation")
    X_test, y_test = unpack_split(test, task, "test")

    train_labels = set(y_train)
    val_labels = set(y_val)
    if train_labels != val_labels:
        raise ValueError(
            f"{task}: train and validation labels differ: "
            f"{train_labels!r} vs {val_labels!r}"
        )

    best_model = None
    best_settings = None
    best_val_f1 = -1.0
    tuning_records = []

    for ngram_range in nRange:
        for c_value in cValues:
            model = make_model(ngram_range, c_value)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            val_metrics = metrics(y_val, predictions)
            tuning_records.append({
                "ngram_range": list(ngram_range),
                "C": c_value,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            })
            print(
                f"{task}: ngrams={ngram_range}, C={c_value}, "
                f"validation macro-F1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_model = model
                best_settings = {
                    "ngram_range": list(ngram_range),
                    "C": c_value,
                }

    test_predictions = best_model.predict(X_test)
    test_metrics = metrics(y_test, test_predictions)

    result = {
        "task": task,
        "model": modelName,
        "split": "test",
        "n": len(y_test),
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }
    append_result(result)

    append_run({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "model": modelName,
        "seed": SEED,
        "train_n": len(y_train),
        "validation_n": len(y_val),
        "test_n": len(y_test),
        "ngram_range": best_settings["ngram_range"],
        "C": best_settings["C"],
        "analyzer": "char",
        "lowercase": True,
        "sublinear_tf": True,
        "max_iter": 2000,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "all_validation_trials": tuning_records,
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
    })

    print(f"\n{task} selected settings: {best_settings}")
    print(f"{task} test metrics: {test_metrics}")
    return result

#Need to run main like this for the file stuff
def main():
    print("Running LD baseline...")
    run_task("LD", load_ld())
    print("\nRunning sentiment baseline...")
    run_task("sentiment", load_sentiment())
    print(f"\nResults appended to: {baselines_path}")
    print(f"Run logs appended to: {runPath}")

if __name__ == "__main__":
    main()

#To run use this in rootDir: python -m ld_sentiment_baseline.py