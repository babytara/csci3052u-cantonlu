import json
from pathlib import Path

predictionsPath = Path(__file__).resolve().parents[1] / "results" / "baseline_test_preds.jsonl"


def plain(value):
    return value.item() if hasattr(value, "item") else value


def save_predictions(task, model, gold, pred):
    predictionsPath.parent.mkdir(parents=True, exist_ok=True)
    with predictionsPath.open("a", encoding="utf-8") as f:
        for i, (g, p) in enumerate(zip(gold, pred)):
            f.write(json.dumps({
                "task": task, "model": model, "split": "test", "index": i,
                "gold": plain(g), "pred": plain(p),
            }, ensure_ascii=False) + "\n")
