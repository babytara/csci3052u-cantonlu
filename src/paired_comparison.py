# Uncertainty and Paired Comparison of Models

# Run from the repository root:
#     python -m src.paired_comparison
 
# This comparison uses predictions for each example from the baseline and llm json files with task, gold, pred, and model
# It then computes the uncertainty with a 95% bootstrap interval for each model and saves it into uncertainty.csv
# The paired comparison compares models against each other using the same examples and in the same order, then saves results into paired_comparison.csv
# McNemar’s test was used as a model comparison test for the examples using paired binary pass and fail data
# The verdict represents the macro-F1 interval, and the accuracy_verdict represents the McNemar test

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from scipy.stats import binomtest
import argparse

# directory to save the results
rootDir = Path(__file__).resolve().parents[1]
resultsDir = rootDir / "results"

# number of bootstrap samples
bootSamples = 10000

SEED = 42

# the LLM to be compared with
LLM = "claude-haiku-4-5"

# loads prediction results
def load(path, default_model = None):
    df = pd.read_json(path, lines = True) # converts prediction results into a dataframe
    # if model name wasn't missing, then assign it
    if "model" not in df:
        df["model"] = default_model

    # returns the task, the model that predicted, the correct answer. and the model's predicted answer 
    return df[["task", "model", "gold", "pred"]]


# avgs each class's f1 scores, using macro-f1 for integer labels.
def macro_f1(gold, pred, classCount):
    f1Scores = []
    # loops through each class
    for c in range(classCount):
        # calculates true positives (model predicted this class, and the actual label is this class)
        truePositive = ((gold == c) & (pred == c)).sum(-1)

        # calculates false positives (model predicted this class, but the actual label isn't this class)
        falsePositive = ((gold != c) & (pred == c)).sum(-1)

        # calculates false negatives (model didn't predict this class, but the actual label is this class)
        falseNegative = ((gold == c) & (pred != c)).sum(-1)
        f1Denominator = 2 * truePositive + falsePositive + falseNegative
        # Calculates the f1 scores, and handles classes that have no labels or predictions
        f1Scores.append(np.where(f1Denominator == 0, 0.0, 2 * truePositive / np.maximum(f1Denominator, 1)))
    return np.mean(f1Scores, axis=0)

def main():
    # Arguments, so it can run on test set too
    ap = argparse.ArgumentParser(description="Bootstrap intervals with paired comparisons on the predictions")
    ap.add_argument("--baseline", default=str(resultsDir / "baseline_preds.jsonl"))
    ap.add_argument("--llm", default=str(resultsDir / "llm_preds.jsonl"))
    ap.add_argument("--split", default="val_subset", help="label added to output files")
    ap.add_argument("--out-dir", default=str(resultsDir))
    args = ap.parse_args()

    # loads the baseline and LLM predictions into a combined dataframe
    df = pd.concat([load(args.baseline), load(args.llm, LLM)], ignore_index=True)
    rng = np.random.default_rng(SEED) # random number generator
    unc_rows = [] # holds the performance metrics and confidence intervals for each model
    pair_rows = [] # for the comparisons between pairs of models

    # organizes data by task
    for task, tdf in df.groupby("task", sort=False):
        # separates the models
        models = {}
        for m, g in tdf.groupby("model", sort=False):
            models[m] = g.reset_index(drop=True)

        # check if every model has the same correct labels
        golds = []
        for g in models.values():
            golds.append(g["gold"].to_list())

        # checks if the same examples and order were used to evaluate each model
        assert all(x == golds[0] for x in golds), f"{task}: models were not scored using the same examples, on the same row, in the same order"

        # changes class labels to integer values
        labels = sorted(set(golds[0]))
        # maps each class to an integer
        code = {}
        for i, l in enumerate(labels):
            code[l] = i
        k = len(labels)

        # changes gold labels and predictions into numpy arrays
        gold = np.array([code[x] for x in golds[0]])
        n = len(gold)

        # changes model predictions to integer arrays
        pred = {m: np.array([code.get(x, -1) for x in g["pred"]]) for m, g in models.items()}

        # gets random bootstrap samples and paired differences
        idx = rng.integers(0, n, (bootSamples, n))  
        boot_f1 = {m: macro_f1(gold[idx], p[idx], k) for m, p in pred.items()} # gets macroF1 for each bootstrap sample
        boot_acc = {m: (gold[idx] == p[idx]).mean(1) for m, p in pred.items()} # gets accuracy for each bootstrap sample
        
        # gets the performance of each model
        for m, p in pred.items():
            f1 = macro_f1(gold, p, k) # f1 from dataset
            acc = float((gold == p).mean()) # accuracy from dataset
            f1Lo, f1Hi = np.percentile(boot_f1[m], [2.5, 97.5]) # upper and lower bounds for macrof1
            accLo, accHi = np.percentile(boot_acc[m], [2.5, 97.5]) # upper and lower bounds for accuracy

            # the uncertainty results of the model
            unc_rows.append(
                dict(
                    task=task,
                    split=args.split,
                    model=m,
                    n=n,
                    accuracy=round(acc, 4),
                    acc_ci_low=round(accLo, 4),
                    acc_ci_high=round(accHi, 4),
                    macro_f1=round(float(f1), 4),
                    f1_ci_low=round(f1Lo, 4),
                    f1_ci_high=round(f1Hi, 4)
                )
            )

        # compares the models in pairs majority, tfidf, LLM
        for a, b in combinations(sorted(pred, key=lambda m: ("majority" != m, m == LLM, m)), 2):
            # calculates the difference in macrof1 between models
            modelF1Difference = float(macro_f1(gold, pred[b], k) - macro_f1(gold, pred[a], k))
            # calculates the bootstrap confidence interval for the difference
            differenceBoot = boot_f1[b] - boot_f1[a]
            diffLo, diffHi = np.percentile(differenceBoot, [2.5, 97.5]) # upper and lower bounds

            # calculates the difference in accuracy between models
            modelAccuracyDifference = float((gold == pred[b]).mean() - (gold == pred[a]).mean())

            # count of model B being correct, but model A being wrong
            onlyBCorrect = int(((pred[b] == gold) & (pred[a] != gold)).sum())
            # count of model A being correct, but model B being wrong
            onlyACorrect = int(((pred[a] == gold) & (pred[b] != gold)).sum())

             # McNemar test
            if onlyACorrect + onlyBCorrect:
                # binomial test p value
                p = binomtest(min(onlyACorrect, onlyBCorrect), onlyACorrect + onlyBCorrect, p=0.5).pvalue
            else:
                p = 1.0

            # mcnemar paired comparison results
            pair_rows.append(
                dict(
                    task=task,
                    split=args.split,
                    n=n,
                    model_a=a,
                    model_b=b,
                    delta_macro_f1=round(modelF1Difference, 4),
                    delta_ci_low=round(diffLo, 4),
                    delta_ci_high=round(diffHi, 4),
                    delta_accuracy=round(modelAccuracyDifference, 4),
                    a_only_correct=onlyACorrect,
                    b_only_correct=onlyBCorrect,
                    mcnemar_p=round(min(p, 1.0), 4),
                    # either real (can distinguish performance of models) or inconclusive (insufficient confirmation to distinguish performance of models)
                    verdict="real" if diffLo > 0 or diffHi < 0 else "inconclusive",
                    accuracy_verdict="real" if p < 0.05 else "inconclusive",
                )
            )

    # outputs results
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unc_rows).to_csv(out / "uncertainty.csv", index=False, lineterminator="\n")
    pd.DataFrame(pair_rows).to_csv(out / "paired_comparison.csv", index=False, lineterminator="\n")
    print(pd.DataFrame(unc_rows).to_string(index=False))
    print()
    print(pd.DataFrame(pair_rows).to_string(index=False))
 
 
if __name__ == "__main__":
    main()
