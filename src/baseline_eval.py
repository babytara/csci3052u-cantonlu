#Majority-class and TF-IDF + logistic regression baselines, scored on the

#Run from the project root:  python -m src.baselines_eval

import csv, json, random
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from src.data import load_ld, load_sentiment, load_nli, stratified_sample


# Same subset settings as llm_eval.py
SEED, N = 42, 100
MAX_TRAIN = None
RESULTS = Path(__file__).resolve().parents[1] / "results"

TASKS = {
    "LD": (load_ld, ["Cantonese", "Mandarin", "corrupted"]),
    "sentiment": (load_sentiment, ["smile", "ok", "cry"]),
    "NLI": (load_nli, ["entailment", "not_entailment"]),
}


def gold_label(row, names):
    g = row[-1]
    return names[g] if isinstance(g, int) else g


def to_text(row):
    return " [SEP] ".join(str(x) for x in row[:-1])


def run(task):
    loader, names = TASKS[task]
    train, val, _test = loader()  # every loader in data.py returns (train, val, test)
    if MAX_TRAIN and len(train) > MAX_TRAIN:
        train = random.Random(SEED).sample(train, MAX_TRAIN)


    sample = stratified_sample(val, N, seed=SEED)

    train_text = [to_text(r) for r in train]
    train_gold = [gold_label(r, names) for r in train]
    test_text = [to_text(r) for r in sample]
    gold = [gold_label(r, names) for r in sample]

    majority = Counter(train_gold).most_common(1)[0][0]
    preds = {"majority": [majority] * len(sample)}

    tfidf_lr = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3)),
        LogisticRegression(max_iter=1000),
    )
    tfidf_lr.fit(train_text, train_gold)
    preds["tfidf_lr"] = list(tfidf_lr.predict(test_text))

    llm_file = RESULTS / "llm_preds.jsonl"
    if llm_file.exists():
        llm_gold = [d["gold"] for d in map(json.loads, llm_file.open(encoding="utf-8"))
                    if d["task"] == task]
        if llm_gold and llm_gold != gold:
            print(f"WARNING: {task} subset differs from llm_preds.jsonl")

    summary, pred_rows = [], []
    for model, pred in preds.items():
        summary.append({
            "task": task, "model": model, "split": "val_subset", "n": len(gold),
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        })
        for text, g, p in zip(test_text, gold, pred):
            pred_rows.append({"task": task, "model": model, "input": text, "gold": g, "pred": p})
    return summary, pred_rows

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    all_summary, all_preds = [], []
    for t in TASKS:
        s, p = run(t)
        for row in s:
            print(row)
        all_summary += s
        all_preds += p
 
    with (RESULTS / "baseline_val.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "split", "n", "accuracy", "macro_f1"])
        w.writeheader()
        w.writerows(all_summary)
 
    with (RESULTS / "baseline_preds.jsonl").open("w", encoding="utf-8") as f:
        for x in all_preds:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

