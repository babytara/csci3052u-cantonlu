# imports needed 
import json
from pathlib import Path
from src.data import load_ld, load_sentiment, load_nli
from src.data import stratified_sample, DATA_DIR

# output folder: repo_root/data/sample
OUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "sample"
OUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

def write_sample(task, rows):
    path = OUT_DIRECTORY / f"{task}_sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(list(row), f, ensure_ascii=False)
            f.write("\n")

def main():
    # run LD
    ld_train, _, _ = load_ld()
    ld_sample = stratified_sample(ld_train, 100, seed=42)
    write_sample("ld", ld_sample)

    # run sentiment
    sa_train, _, _ = load_sentiment()
    sa_sample = stratified_sample(sa_train, 100, seed=42)
    write_sample("sentiment", sa_sample)

    # run NLI
    nli_train, _, _ = load_nli()
    nli_sample = stratified_sample(nli_train, 100, seed=42)
    write_sample("nli", nli_sample)

if __name__ == "__main__":
    main()
