# data loading
import json
from pathlib import Path
from datasets import load_dataset

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

