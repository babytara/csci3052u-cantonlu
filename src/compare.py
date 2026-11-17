# python -m src.compare
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

RESULTS = Path(__file__).resolve().parents[1] / "results"
N_BOOT, SEED = 2000, 0

def load(path, default_model=None):
    df = pd.read_json(path, lines=True)
    if "model" not in df:
        df["model"] = default_model
    return df[["task", "model", "gold", "pred"]]

base = load(RESULTS / "baseline_preds.jsonl")
llm = load(RESULTS / "llm_preds.jsonl", "claude-haiku-4-5")
df = pd.concat([base, llm], ignore_index=True)

def fmt(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"

rng = np.random.default_rng(SEED)
rows = []
for task, tdf in df.groupby("task", sort=False):
    models = {m: g.reset_index(drop=True) for m, g in tdf.groupby("model", sort=False)}

   
    golds = [g["gold"].tolist() for g in models.values()]
    assert all(x == golds[0] for x in golds), f"{task}: models not scored on the same rows"

    gold = np.array(golds[0])
    n, task_labels = len(gold), sorted(set(gold))
    idxs = rng.integers(0, n, (N_BOOT, n))   

    for m, g in models.items():
        pred = g["pred"].to_numpy()
        acc = [accuracy_score(gold[i], pred[i]) for i in idxs]
        f1 = [f1_score(gold[i], pred[i], labels=task_labels, average="macro", zero_division=0) for i in idxs]
        a_lo, a_hi = np.percentile(acc, [2.5, 97.5])
        f_lo, f_hi = np.percentile(f1, [2.5, 97.5])
        rows.append({
            "task": task, "model": m, "n": n,
            "accuracy": fmt(accuracy_score(gold, pred), a_lo, a_hi),
            "macro_f1": fmt(f1_score(gold, pred, labels=task_labels, average="macro", zero_division=0), f_lo, f_hi),
        })

table = pd.DataFrame(rows)
table.to_csv(RESULTS / "comparison.csv", index=False)

# Markdown Version
print("| " + " | ".join(table.columns) + " |")
print("|" + "---|" * len(table.columns))
for r in table.itertuples(index=False):
    print("| " + " | ".join(map(str, r)) + " |")

# Confusion Matrics
def plot_confusion_matrix(y_true, y_pred, labels, group, out_path=None):
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(f"Confusion Matrix for {group}")
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", np.trace(cm) / cm.sum())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for lab, p_, r_, s_ in zip(labels, p, r, support):
        print(f"{lab}: Precision: {p_:.3f}, Recall: {r_:.3f}, Support: {s_}")

#Highest prediction accuracy sentiment: tfidf_lr from the baseline validation
d = df[(df["task"] == "sentiment") & (df["model"] == "tfidf_lr")]
plot_confusion_matrix(d["gold"], d["pred"], ["smile", "ok", "cry"],
                      "Sentiment - tfidf_lr",
                      RESULTS / "confusion_matrix_sentiment_tfidf_lr.png")

# Lowest macro-F1, LD: majority
d = df[(df["task"] == "LD") & (df["model"] == "majority")]
plot_confusion_matrix(d["gold"], d["pred"], ["Cantonese", "Mandarin", "corrupted"], 
                      "LD - majority",
                      RESULTS / "confusion_matrix_ld_majority.png")

# Highest accuracy, NLI: Claude Haiku
d = df[(df["task"] == "NLI") & (df["model"] == "claude-haiku-4-5")]
plot_confusion_matrix(d["gold"], d["pred"], ["entailment", "not_entailment", "UNPARSED"],
                      "NLI - claude-haiku-4-5",
                      RESULTS / "confusion_matrix_nli_claude_haiku.png")  

