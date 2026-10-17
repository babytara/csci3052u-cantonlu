
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue # skip blank lines
            items.append(json.loads(line))
    return items

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
    
