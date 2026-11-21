# Contributions 
## Milestone 2

| Member Names | Tasks/Contribution |
| --- | ---|
| Gabriel Pedraza Perez | <ul><li>NLI + shared plot style</li><li>Plot Style</li><li>Construction process - Machine-translated from English NLI</li><li>Target/task in one line</li><li>Class balance plot + text length histogram</li><li>Construction process for each (from data/ld/readme.md and data/sentiment/readme.md)</li><li>Write up</li>
| Omotara Suleiman | <ul><li>Language Detection + Sentiment</li><li>Construction process</li><li>Class balance plot + text length histogram</li><li>Cleaning checks: nulls, duplicates, empty strings, unexpected labels</li><li>Write up</li> |
|Olohigbe Uantioje | <ul><li>WSD + LAJ</li><li>Construction process for each (WSD manually compiled; LAJ adapted from SiniticMTError)</li><li>Note that LAJ ships in two forms — finetune and comparison</li><li>Class balance plot + text length histogram</li><li>Write up</li> |
| Kabishan Sutharman | <ul><li>Report assembly + submission</li><li>Short intro naming the five datasets and the project objective</li><li>Compile all four deliverables into one PDF, in the announcement's order </li><li>Licence statement: repo says MIT, Hyun Min's email said CC-BY — state both</li><li>Append the GitHub URL at the end</li><li>Confirm repo is accessible to instructors and TAs</li><li>Fix the proposal for consistency: sentiment labels are smile/ok/cry, WSD/LAJ now in scope</li> |
|Jin Sutharman | <ul><li>Pre-processing (all five tasks)</li><li>Label encoding per task </li><li>Text handling: character n-gram TF-IDF, no word segmentation needed</li><li>Train/validation/test strategy — using the released splits</li><li>Planned LLM sampling: 300 per task, stratified, fixed seed</li><li>Mark clearly which steps are done vs. planned (announcement allows planned)</li> |

## Milestone 3
|Members | Tasks/Contribution |
| --- | --- |
| Gabriel Pedraza Perez | TF-IDF character n-grams + logistic regression for both tasks. Tune on val only, score once on test. Results to results/baselines.csv. Log each run (seed, settings) and record them for the memo. |
| Omotara Suleiman | Loaders for LD, sentiment, NLI returning train/val/test. NLI download script (hon9kon9ize/yue-all-nli). Fixed seed + stratified-sample function. Pinned requirements.txt with versions, README "run from scratch" with exact commands, no hard-coded personal paths, small sample file so the pipeline runs without the full data, fresh-clone test. |
| Olohigbe Uantioje | 2-page memo from A–D (dataset commit, split, preprocessing, seeds, hyperparameters, software versions, commands), PDF export. Tag release v0.3-baseline. Update AI_USE.md and CONTRIBUTIONS.md. Final check against the spec. |
| Kabishan Sutharman | Majority-class predictor for all 3 tasks (accuracy + macro-F1 on test). TF-IDF + logistic regression for NLI. Results to results/baselines.csv. Log each run (seed, settings). |
| Jin Sutharman | Baseline table from baselines.csv plus the paper’s reported numbers (cite page). Explain differences: model class, label noise, class imbalance, metric, sample vs full test set. Start/stop/continue plan for Milestones 4–7. Script that regenerates the table from baselines.csv. |

## Milestone 4
|Members | Tasks/Contribution |
| --- | --- |
| Gabriel Pedraza Perez | Plan for Milestone 5 (Nov 20) and Milestone 6 (Dec 4), who does what, and the midpoint peer check-in. |
| Omotara Suleiman |Models in use: majority class, character TF-IDF + logistic regression, commercial LLMs. One equation each for logistic regression (loss and regularization C), TF-IDF, and macro-F1. |
| Olohigbe Uantioje | What changed since the proposal, what is stuck, and how we responded to instructor feedback from Milestones 1–3. |
| Kabishan Sutharman | Title slide. The task (CantoNLU: language detection, sentiment, NLI), why it matters, the data pipeline and splits, and the primary metric.|
| Jin Sutharman | Baseline vs LLM table on the same subset, confidence intervals, and one error finding (confusion matrix or example failures).|

**Support Task**
|Members | Support Task | Work |
| --- | --- | --- |
| Gabriel Pedraza Perez | Progress Report | Methods and theory, preliminary results, deviations, blockers, revised timeline, response to feedback. |
| Omotara Suleiman | LLM runs | Pick the model and prompt, write the script that sends each example and maps replies to labels, run it on the validation subset, log model, prompt and cost to results/. |
| Olohigbe Uantioje | Baseline rescoring and analysis | Rescore majority class and TF-IDF + logistic regression on the same subset, add confidence intervals, build the comparison table, make one confusion matrix and note the main error pattern.|
| Kabishan Sutharman | Deck and Recording | Merge everyone’s slides into one template, run the recording call, export the video, check audio. |
| Jin Sutharman | Release and record | Tag v0.4-progress, update AI_USE.md and CONTRIBUTIONS.md, collect everyone’s peer check-in confirmation.|

## Milestone 5
|Members | Block | Tasks |
| --- | --- | --- |
| Gabriel Pedraza Perez | Baselines: rescore and Tune | Rescore the baselines on the same 100 validation Extend the C grid past 10. Make ld_sentiment_baseline.py skip logged rows. Use one Python and scikit-learn version for every row. Write down the search grid, the chosen values and the selection rule. Save per-example predictions for every baseline run. |
| Omotara Suleiman | LLM runs | Estimate cost before any run. Run the LD prompt with and without the “corrupted” definition on validation (this is also the ablation). Score the test set once. Keep llm_cache.jsonl and llm_preds.jsonl up to date. |
| Olohigbe Uantioje | Error analysis and robustness | Confusion matrices from llm_preds.jsonl and the baseline predictions. Explain the LD Mandarin over-prediction and the sentiment “ok” class. One sensitivity check, one robustness check and one ablation. |
| Kabishan Sutharman | Uncertainty and Comparison | Bootstrap 95% confidence intervals for macro-F1. Paired comparison of each model against the baselines on the same examples. State which gaps are within noise. |
| Jin Sutharman | Writing and records | Draft the Results section. Build the main table or figure. Keep the experiment log. Compare the draft against the instructor’s sample when it arrives. Update the README, AI_USE.md and CONTRIBUTIONS.md. Tag the release. |

**Support Task**
| Members | Support Task | Work Done |
| --- | --- | --- |
| Gabriel Pedraza Perez | Experiment Log | One row per run: date, model, task, split, configuration, hyperparameters, environment, result file, observations|
| Omotara Suleiman | Results Package assembly | Result CSVs, predictions, figures and the log are in one folder in the format Canvas asks for|
| Olohigbe Uantioje | AI_USE.md and CONTRIBUTIONS.md entries | Update AI use and contribution markdown |
| Kabishan Sutarman | Release tag and README check | ag exists and every README table matches results/*.csv |
