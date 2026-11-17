import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, f1_score
from src.data import load_ld, stratified_sample

from src.llm_eval import (
    ask, parse, TASKS, load_cache,
    CACHE, MODEL, RESULTS, SEED, N
)

COST_PER_CALL = 0.0003

# LD prompt without the corrupted label definition
def ld_no_definition(row):
    return (
        "This sentence is written in Chinese. Your role is to act as an expert "
        "language identifier. Classify which category it belongs to: Cantonese, "
        "Mandarin, or corrupted. Answer with one word.\n\n"
        "Cantonese, Mandarin, corrupted.\n\n"
        f"Sentence: {row[0]}"
    )



def cache_key(task, prompt):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return f"{MODEL}|{task}|{prompt_hash}"


def evaluate(task, rows, make_prompt, names, split, variant, yes):
    cache = load_cache()

    prompts = [make_prompt(row) for row in rows]
    # find prompts that are not already cached
    missing = []
    seen = set()

    for prompt in prompts:
        key = cache_key(task, prompt)
        if key not in cache and prompt not in seen:
            missing.append(prompt)
            seen.add(prompt)

    estimated_cost = len(missing) * COST_PER_CALL
    print(f"{task}/{variant}: {len(missing)} new calls, about ${estimated_cost:.2f}")

    # ask before making API calls
    if missing and not yes:
        if input("continue? (yes/no) ") != "yes":
            sys.exit("stopped!")

    # run missing prompts and save each response to the cache
    with ThreadPoolExecutor(max_workers=3) as executor:
        with CACHE.open("a", encoding="utf-8") as file:
            for count, (prompt, reply) in enumerate(
                zip(missing, executor.map(ask, missing)), start=1
            ):
                key = cache_key(task, prompt)
                cache[key] = reply
                entry = {"key": key, "reply": reply}
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                file.flush()

                if count % 500 == 0:
                    print(f"{task}: {count}/{len(missing)}", flush=True)

    # collect predictions and labels
    gold = []
    predictions = []
    output = []
    unparsed = 0

    for index, row in enumerate(rows):
        prompt = make_prompt(row)
        reply = cache[cache_key(task, prompt)]
        prediction = parse(reply, names)

        if prediction is None:
            unparsed += 1
        actual = row[-1]
        if isinstance(actual, int):
            actual = names[actual]

        gold.append(actual)
        predictions.append(prediction or "UNPARSED")

        output.append({
            "task": task,
            "variant": variant,
            "idx": index,
            "input": list(row[:-1]),
            "gold": actual,
            "pred": predictions[-1]
        })

    # calculate the eval metrics
    result = {
        "task": task,
        "model": MODEL,
        "split": split,
        "variant": variant,
        "n": len(gold),
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(
            gold, predictions, labels=names, average="macro", zero_division=0
        ),
        "unparsed": unparsed
    }

    return result, output

# replace last results for the same task and variant
def save_results(csv_path, jsonl_path, result, output):
    previous = []

    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as file:
            previous = list(csv.DictReader(file))

    previous = [
        row for row in previous
        if (row["task"], row["variant"]) !=
           (result["task"], result["variant"])
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerows(previous + [result])

    # keep predictions from other tasks and variants
    previous_lines = []

    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if (record["task"], record["variant"]) != (
                    result["task"], result["variant"]
                ):
                    previous_lines.append(line)

    with jsonl_path.open("w", encoding="utf-8") as file:
        file.writelines(previous_lines)

        for record in output:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    yes = "--yes" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--yes"]
    mode = args[0]

    RESULTS.mkdir(exist_ok=True)

    if mode == "ablation":
        loader, names, make_prompt = TASKS["LD"]
        rows = stratified_sample(load_ld()[1], N, seed=SEED)

        variants = [
            ("with_definition", make_prompt),
            ("no_definition", ld_no_definition)
        ]

        for variant, prompt_fn in variants:
            result, output = evaluate(
                "LD", rows, prompt_fn, names,
                "val_subset", variant, yes
            )

            print(result)
            save_results(
                RESULTS / "llm_ablation.csv",
                RESULTS / "llm_ablation_preds.jsonl",
                result,
                output
            )

    elif mode == "test":
        tasks = args[1:] or list(TASKS)

        for task in tasks:
            loader, names, make_prompt = TASKS[task]

            result, output = evaluate(
                task, loader()[2], make_prompt,
                names, "test", "definition", yes
            )

            print(result)
            save_results(
                RESULTS / "llm_test.csv",
                RESULTS / "llm_test_preds.jsonl",
                result,
                output
            )