# Pairwise co-use features for flagging dangerous polypharmacy

Motivation: The paper "Neural bandits for data mining: searching for dangerous polypharmacy" (2023) searches for dangerous combinations of substances with a learned reward model. This experiment asks whether a simpler, explicitly interpretable feature set can flag the same kind of risk more accurately, using a neural-network surrogate as the reference point.

Question: Can a class-balanced LightGBM model with pairwise co-use features flag dangerous polypharmacy better than a standard MLP surrogate that sees only the raw per-substance levels, on a public substance-use dataset with a rule-based danger label?

Dataset: UCI Drug Consumption (quantified), 1,885 people with 18 substance-use levels (0-6), loaded through the `ucimlrepo` package (id 373). This is not the paper's own dataset. "Dangerous" is a rule-based stand-in label (polypharmacy plus an a-priori sedation-risk pair), not clinical truth.

Baseline vs improved model:
- Baseline: standardised MLP (64-32 hidden units) on the raw substance levels. It stands in for the paper's neural reward model and is not the neural bandit itself.
- Improved: class-balanced LightGBM on the raw levels, a total-count feature, and all 153 pairwise co-use indicators.

Both models use the same single stratified 70/30 split, the same CV-based tuning procedure, and PR-AUC as the main metric, with F1 and ROC-AUC as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts in `figures/`).

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
