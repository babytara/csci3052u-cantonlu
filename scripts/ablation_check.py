import sys
import json
from collections import Counter
from scipy.stats import binomtest

# load all predictions
with open("results/llm_ablation_preds.jsonl", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

# group by prompt variant
with_def = {
    row["idx"]: row
    for row in data
    if row["variant"] == "with_definition"
}

without_def = {
    row["idx"]: row
    for row in data
    if row["variant"] == "no_definition"
}

# count predicted labels for each 
for variant, predictions in [
    ("with_definition", with_def),
    ("no_definition", without_def)
]:
    label_counts = Counter(row["pred"] for row in predictions.values())
    print(variant, label_counts)

# compare cases where only one variant predicted correctly
with_correct = 0
without_correct = 0

for idx in with_def:
    gold = with_def[idx]["gold"]
    is_with_correct = with_def[idx]["pred"] == gold
    is_without_correct = without_def[idx]["pred"] == gold
    if is_with_correct and not is_without_correct:
        with_correct += 1
    elif is_without_correct and not is_with_correct:
        without_correct += 1

print("only with:", with_correct)
print("only without:", without_correct)

# binomial significance test
total = with_correct + without_correct
if total == 0:
    sys.exit("the prompts never disagreed")
p_value = binomtest(with_correct, total).pvalue
print("p-value:", p_value)