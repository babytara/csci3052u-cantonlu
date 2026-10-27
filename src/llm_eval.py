import csv, hashlib, json, os
import time 
import anthropic
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import accuracy_score, f1_score
# data loaders
from src.data import load_ld, load_sentiment, load_nli, stratified_sample

SEED, N = 42, 100 
MODEL = "claude-haiku-4-5-20251001"  # model being used for evaluation

# results and cache goes here
RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = RESULTS / "llm_cache.jsonl"

_client = None

# prompts for LLMs
# prompts for LLMs
def ask(prompt):
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=8)
    r = _client.messages.create(
        model=MODEL, max_tokens=20,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")

TASKS = {
    "LD": (
        load_ld,
        ["Cantonese", "Mandarin", "corrupted"],
        lambda r: (
            "This sentence is written in Chinese. Your role is to act as an expert "
            "language identifier. Classify which category it belongs to: Cantonese, "
            "Mandarin, or corrupted (a sentence where some words were swapped for the other "
            "language's equivalents, so it partly mixes Cantonese and Mandarin). Answer with one word.\n\n"
            "Cantonese, Mandarin, corrupted.\n\n"
            f"Sentence: {r[0]}"
        ),
    ),

    "sentiment": (
        load_sentiment,
        ["smile", "ok", "cry"],
        lambda r: (
            "Your task is to act as an expert and identify the sentiment of this Cantonese sentence. "
            "Pick one of the following labels: smile (positive), ok (neutral), "
            "or cry (negative). Answer with one word.\n\n"
            f"Sentence: {r[0]}"
        ),
    ),

    "NLI": (
        load_nli,
        ["entailment", "not_entailment"],
        lambda r: (
            "You are given a premise and a hypothesis. Your task is to decide whether "
            "the premise supports/ entails the hypothesis. Answer with one word: "
            "entailment or not_entailment.\n\n"
            f"Premise: {r[0]}\n"
            f"Hypothesis: {r[1]}"
        ),
    ),
}

# extract labels
def parse(reply, names):
    reply = reply.strip().lower()
    for n in sorted(names, key=len, reverse=True):
        if n.lower() in reply:
            return n
    return None

# load cache from disk
def load_cache():
    if not CACHE.exists():
        return {}
    with CACHE.open(encoding="utf-8") as f:
        return {d["key"]: d["reply"] for d in map(json.loads, f)}

# run evaluation for a task
def run(task):
    loader, names, make_prompt = TASKS[task]
    sample = stratified_sample(loader()[1], N, seed=SEED) 
    cache = load_cache()
    gold = []
    pred = []
    unparsed = 0
    rows = []

    # cache key for a prompt
    def make_key(prompt):
        return f"{MODEL}|{task}|{hashlib.md5(prompt.encode()).hexdigest()}"

    # only ask the model for prompts we have not answered yet
    prompts = [make_prompt(row) for row in sample]
    todo = [p for p in prompts if make_key(p) not in cache]

    # send 3 requests at a time to speed things up
    RESULTS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for p, reply in zip(todo, ex.map(ask, todo)):
            cache[make_key(p)] = reply
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": make_key(p), "reply": reply}, ensure_ascii=False) + "\n")

    # score every example from the cache
    for row in sample:
        prompt = make_prompt(row)
        label = parse(cache[make_key(prompt)], names)
        unparsed += label is None

        g = row[-1] # gold label
        gold.append(names[g] if isinstance(g, int) else g)
        pred.append(label or "UNPARSED") # predicted label 

        rows.append({
            "task": task,
            "input": list(row[:-1]),
            "gold": gold[-1],
            "pred": pred[-1],
        })

    # write results to jsonl 
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "llm_preds.jsonl").open("a", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    return {"task": task, "model": MODEL, "split": "val_subset", "n": len(gold),
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
            "unparsed": unparsed}

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    # fresh start every run
    (RESULTS / "llm_preds.jsonl").unlink(missing_ok=True)  
    with (RESULTS / "llm_val.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "split", "n", "accuracy", "macro_f1", "unparsed"])
        w.writeheader()
        for t in TASKS:
            row = run(t)
            print(row)
            w.writerow(row)