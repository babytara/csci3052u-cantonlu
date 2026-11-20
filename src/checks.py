# python -m src.checks
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar   # pip install statsmodels

rootDir = Path(__file__).resolve().parents[1]
resultDir = rootDir / "results"
resultDir.mkdir(parents=True, exist_ok=True)

LABELS = {
    "sentiment": ["smile", "ok", "cry"],
    "LD": ["corrupted", "Mandarin", "Cantonese"],
    "NLI": ["entailment", "not_entailment"],
}

def load(path, default_model=None):
    df = pd.read_json(path, lines=True)
    if "model" not in df:
        df["model"] = default_model
    if "variant" not in df:
        df["variant"] = None
    keep = [c for c in ["task", "model", "variant", "idx", "input", "gold", "pred"] if c in df]
    return df[keep]

def make_key(x):
    return " || ".join(x) if isinstance(x, list) else str(x)

def score(g):
    labels = sorted(set(g["gold"]))
    return {"n": len(g),
            "acc": accuracy_score(g["gold"], g["pred"]),
            "macro_f1": f1_score(g["gold"], g["pred"], labels=labels,
                                 average="macro", zero_division=0)}


# ---------- 1. label-noise robustness ----------
def label_noise(df):
    print("\n\n------label-noise robustness------")
    df = df.copy()
    df["key"] = df["input"].map(make_key)
    grp = df.groupby(["task", "model", "key"])["gold"]
    df["is_dup"] = grp.transform("size") > 1
    df["is_conflict"] = grp.transform("nunique") > 1
    # Duplicate / conflict counts
    print("rows in duplicated groups:", int(df["is_dup"].sum()))
    print("rows in conflicting groups:", int(df["is_conflict"].sum()))
    print("distinct duplicated inputs:", df[df["is_dup"]].groupby(["task", "model", "key"]).ngroups)
    print("distinct conflicting inputs:", df[df["is_conflict"]].groupby(["task", "model", "key"]).ngroups)
    print(df.groupby(["task", "model"])[["is_dup", "is_conflict"]].sum())

    subsets = {
        "all": df,
        "dedup": df.drop_duplicates(["task", "model", "key", "gold"]),
        "no_conflict": df[~df["is_conflict"]],
        "unique_only": df[~df["is_dup"]],
    }
    rows = []
    for name, s in subsets.items():
        for (task, model), g in s.groupby(["task", "model"]):
            rows.append({"subset": name, "task": task, "model": model, **score(g)})
    res = pd.DataFrame(rows)
    res.to_csv(resultDir / "label_noise.csv", index=False)
    print(res.pivot_table(index=["task", "model"], columns="subset", values="acc").round(3))


# ---------- 2. sensitivity ----------
def sensitivity(llm):
    print("\n\n------sensitivity to prompt variants------")
    rows = [{"task": t, "variant": v, **score(g)} for (t, v), g in llm.groupby(["task", "variant"])]
    res = pd.DataFrame(rows)
    res.to_csv(resultDir / "variant_sensitivity.csv", index=False)
    print(res.round(3))

    for task, g in llm.groupby("task"):
        rate = (g["pred"] == "UNPARSED").mean()
        kept = g[g["pred"] != "UNPARSED"]
        print(task, "unparsed:", round(rate, 3),
              "acc as-wrong:", round(accuracy_score(g["gold"], g["pred"]), 3),
              "acc dropped:", round(accuracy_score(kept["gold"], kept["pred"]), 3))
    # Markdown Version
    print("| " + " | ".join(res.columns) + " |")
    print("|" + "---|" * len(res.columns))
    for r in res.itertuples(index=False):
        print("| " + " | ".join(map(str, r)) + " |")


# ---------- 3. ablation ----------
def ablation(abl):
    print("\n\n------ablation study------")
    ld = abl[abl["task"] == "LD"]
    w = ld[ld["variant"] == "with_definition"].set_index("idx").sort_index()
    n = ld[ld["variant"] == "no_definition"].set_index("idx").sort_index()
    assert w.index.equals(n.index) and (w["gold"] == n["gold"]).all(), "variants not on same rows"

    cw = (w["pred"] == w["gold"]).to_numpy()
    cn = (n["pred"] == n["gold"]).to_numpy()
    print("acc with:", cw.mean(), "without:", cn.mean())
    print(mcnemar([[(cw & cn).sum(), (cw & ~cn).sum()],
                   [(~cw & cn).sum(), (~cw & ~cn).sum()]], exact=True))

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(cw), (2000, len(cw)))
    diff = cw[idx].mean(axis=1) - cn[idx].mean(axis=1)
    print("diff:", diff.mean(), np.percentile(diff, [2.5, 97.5]))

    table = pd.DataFrame([{"task": "LD", "n": len(cw),
                           "acc_with": cw.mean(), "acc_without": cn.mean(),
                           "diff": diff.mean(),
                           "diff_ci_low": np.percentile(diff, 2.5),
                           "diff_ci_high": np.percentile(diff, 97.5)}])
    table.to_csv(resultDir / "ablation.csv", index=False)
    
    
    # Markdown Version
    print("| " + " | ".join(table.columns) + " |")
    print("|" + "---|" * len(table.columns))
    for r in table.itertuples(index=False):
        print("| " + " | ".join(map(str, r)) + " |")


if __name__ == "__main__":
    base = load(resultDir / "baseline_preds.jsonl")
    llm = load(resultDir / "llm_test_preds.jsonl", "claude-haiku-4-5")
    abl = load(resultDir / "llm_ablation_preds.jsonl", "claude-haiku-4-5")

    label_noise(pd.concat([llm, base], ignore_index=True))
    sensitivity(llm)
    ablation(abl)
