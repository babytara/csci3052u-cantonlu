import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

rootDr = Path(__file__).resolve().parents[1]
Results = rootDr / "results"
matrixDir = Results / "confusion_matrix"

def load(path, default_model=None):
    df = pd.read_json(path, lines=True)
    if "model" not in df:
        df["model"] = default_model
    return df [["task", "model","gold", "pred"]]

base = load(Results/"baseline_preds.jsonl")
llm = load(Results/"llm_test_preds.jsonl", "claude-haiku-4-5")
df = pd.concat([llm, base], ignore_index=True)

def plot_confusion_matrix(y_true, y_pred, labels, group, out_path = None):
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
    disp.plot(cmap ="Blues", values_format=".2f", xticks_rotation=45)
    plt.title(f"Confusion Matrix for {group}")
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    
    print(f"Confusion matrix for {group}:")
    print("Accuracy: ", np.trace(cm) / cm.sum())
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for lab, p_, r_, s_ in zip(labels, p, r, support):
        print(f"{lab}: Precision: {p_:.3f}, Recall: {r_:.3f}, Support: {s_}")

#LLM Test Predictions        
d = df[(df["task"] == "sentiment") & (df["model"] == "claude-haiku-4-5")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["smile","ok", "cry"],
                        group="sentiment - test", out_path=matrixDir/"llm_test_sentiment_confusion_matrix.png")

d = df[(df["task"] == "LD") & (df["model"] == "claude-haiku-4-5")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["corrupted", "Mandarin", "Cantonese"],
                        group="LD - test", out_path=matrixDir/"llm_test_LD_confusion_matrix.png")

d = df[(df["task"] == "NLI") & (df["model"] == "claude-haiku-4-5")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["entailment", "not_entailment"],
                        group="NLI - test", out_path=matrixDir/"llm_test_NLI_confusion_matrix.png")

## ------- Baseline Predictions ---------
#TFIDF with Linear Regression
d = df[(df["task"] == "sentiment") & (df["model"] == "tfidf_lr")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["smile","ok", "cry"],
                        group="sentiment - baseline/tfidf", out_path=matrixDir/"baseline_sentiment_tfidf_confusion_matrix.png")

d = df[(df["task"] == "NLI") & (df["model"] == "tfidf_lr")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["entailment", "not_entailment"],
                        group="NLI - baseline/tfidf", out_path=matrixDir/"baseline_NLI_tfidf_confusion_matrix.png")

d = df[(df["task"] == "LD") & (df["model"] == "tfidf_lr")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["corrupted", "Mandarin", "Cantonese"],
                        group="LD - baseline/tfidf_lr", out_path=matrixDir/"baseline_LD_tfidf_lr_confusion_matrix.png")

# Majority Baseline
d = df[(df["task"] == "LD") & (df["model"] == "majority")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["corrupted", "Mandarin", "Cantonese"],
                        group="LD - baseline/majority", out_path=matrixDir/"baseline_LD_majority_confusion_matrix.png")

d = df[(df["task"] == "sentiment") & (df["model"] == "majority")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["smile", "ok", "cry"],
                        group="sentiment - baseline/majority", out_path=matrixDir/"baseline_sentiment_majority_confusion_matrix.png")

d = df[(df["task"] == "NLI") & (df["model"] == "majority")]
plot_confusion_matrix(d["gold"], d["pred"], labels=["entailment", "not_entailment"],
                        group="NLI - baseline/majority", out_path=matrixDir/"baseline_NLI_majority_confusion_matrix.png")

