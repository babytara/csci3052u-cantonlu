import shutil
from pathlib import Path
out = Path("results_package"); out.mkdir(exist_ok=True)
for pat in ["results/*.csv", "results/*_preds.jsonl", "results/runs.jsonl", "results/*.png"]:
    for f in Path(".").glob(pat):
        shutil.copy(f, out / f.name)
shutil.make_archive("results_package", "zip", out)
print("built results_package.zip")