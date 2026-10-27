# Generative AI Use 
## Tara Suleiman - Milestone 2, LD and Sentiment Analysis
**Tools used:** Claude </br>
**Date:** 2nd October 2026

| Purpose | Material Used | Verification |
|---|---|---|
| More help understanding Python concepts, including JSON Lines parsing, dictionary/list structures, `Counter`, and `setdefault` | Conceptual explanations only, all code was written by me | Ran each cell and reviewed the output to confirm the code worked as expected |
| Reviewing plot style chosen by team | Ask for feedback on the readability of the colour palette | Reviewed the feedback and decided to keep the original colour palette |
| Adjusting the histogram to zoom in on the most relevant data range | Claude suggested using `plt.xlim()` to zoom in on the histogram’s x-axis | Ran the updated code and confirmed that the histogram displayed like I wanted |

**Verification statement:** All code in `notebooks/01_explore_ld_sentiment.ipynb` was written and executed by me. Every figure and statistic in the findings sections is reproduced by a cell in that notebook.

## Tara Suleiman, Milestone 3–4

- **Tool:** Claude (Anthropic), used as a coding and planning 

### Model under evaluation (not an assistant)

- **Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), Anthropic API, run on Oct 27, 2026. Zero-shot prompts are in `src/llm_eval.py`. Temperature was left at default.
- A first trial on Gemini 3.1 Flash-Lite was abandoned because of repeated 503 errors, and its results are not used.