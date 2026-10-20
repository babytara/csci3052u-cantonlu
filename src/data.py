# data loading
import json
import random
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

NLI_REPOSITORY = "hon9kon9ize/yue-all-nli"
NLI_REVISION = "f43c2e709cc3bd448a9c5d5cb5465646cd5f36bb"

# path to the data directory
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "canto-nlu" / "data"

# loaders for the datasets
def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue # skip blank lines
            items.append(json.loads(line))
    return items

# loader for ld dataset
def load_ld():
    splits = ("train", "val", "test")
    data = {
        split: [
            (row["sentence"], row["label"])
            for row in load_jsonl(DATA_DIR / "ld" / f"ld_{split}.jsonl")
        ]
        for split in splits
    }
    return data["train"], data["val"], data["test"]

# loader for sentiment dataset
def load_sentiment():
    splits = ("train", "valid", "test")
    data = {
        split: [
            (row["sentence"], row["label"])
            for row in load_jsonl(DATA_DIR / "sentiment" / f"{split}.jsonl")
        ]
        for split in splits
    }
    return data["train"], data["valid"], data["test"]

# loader for NLI dataset
def load_nli():
    dataset = load_dataset(NLI_REPOSITORY, revision=NLI_REVISION)

    def unroll(split):
        anchors = split["anchor"]
        positives = split["positive"]
        negatives = split["negative"]
        
        pairs = []
        for anchor, positive, negative in zip(anchors, positives, negatives):
            pairs.append((anchor, positive, "entailment"))
            pairs.append((anchor, negative, "not_entailment"))
        return pairs

    train = unroll(dataset["train"])
    valid = unroll(dataset["dev"])
    test = unroll(dataset["test"])

    return train, valid, test

def stratified_sample(data, n, seed=42):
    if n > len(data):
        raise ValueError(f"n - {n} larger than dataset size - ({len(data)})")

    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in data: groups[row[-1]].append(row)
    
    total = len(data)
    # proportional counts for each label
    counts = {label: int(len(rows) / total * n) for label, rows in groups.items()}

    # rounding adjustment 
    missing = n - sum(counts.values())

    if missing:
        largest_label = max(groups, key=lambda lbl: len(groups[lbl]))
        counts[largest_label] += missing

    # sample the data & shuffle
    sampled = [row for label, rows in groups.items() for row in rng.sample(rows, counts[label])]
    rng.shuffle(sampled)
    return sampled

 # load the committed 100 row sample for each task

def load_sample(task):
    path = DATA_DIR.parents[1] / "sample" / f"{task}_sample.jsonl"
    rows = load_jsonl(path)
    return [tuple(row) for row in rows]