# Data

The CantoNLU datasets are not committed to our repository. Clone the source repository to get them.

## How to get the data

From the repository root:

```bash
git clone https://github.com/Aatlantise/canto-nlu.git data/canto-nlu
git -C data/canto-nlu checkout 2283874
```

## Source and licence

- CantoNLU (Min et al., 2025): https://github.com/Aatlantise/canto-nlu, pinned at commit `2283874`.
- The repository is released under MIT. The project team lead confirmed by email that contributions should be treated as CC-BY. Attribute to Min et al. (2025), arXiv:2510.20670.

## NLI data

The full NLI data is not included in the repository. `load_nli()` downloads it from Hugging Face (`hon9kon9ize/yue-all-nli`), pinned to revision `f43c2e709cc3bd448a9c5d5cb5465646cd5f36bb`. The lab repo contains only a 100-row NLI sample.

## Small samples

`data/sample/` contains a committed 100-row sample for each task (LD, sentiment, NLI), drawn from the train split with seed 42. See the main README for how to load and regenerate them.