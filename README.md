README.md unchanged. The method, dataset, baseline, split and metrics are the same; only the column lookup in load_data() was made robust (substance columns are matched by normalised name across the features and targets frames, instead of a direct features['Alcohol'] lookup that raised KeyError). The README was not provided to me, so it has not been edited.

## Results (latest run)

On the UCI Drug Consumption (quantified) dataset, a standard MLP surrogate classifier reached PR-AUC 0.960 for dangerous polypharmacy on the held-out test set, while my class-balanced LightGBM with pairwise co-use features reached 1.00, though those features include the label's own co-use pairs.

| Metric | Value |
|---|---|
| baseline_score | 0.9604 |
| baseline_score_std | 0.01161 |
| improved_score | 1 |
| improved_score_std | 0 |
| improvement | 0.03961 |
| baseline_f1 | 0.8949 |
| baseline_f1_std | 0.02546 |
| improved_f1 | 1 |
| improved_f1_std | 0 |
| baseline_roc_auc | 0.987 |
| baseline_roc_auc_std | 0.003968 |
| improved_roc_auc | 1 |
| improved_roc_auc_std | 0 |
| n_test_positive | 147 |
| test_positive_rate | 0.2597 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in rule-based label, not clinical truth. The improved feature set contains the exact pairwise indicators used to define the label, so the gain partly reflects label construction. Only one train/test split was used (split-once rule), so std covers model seeds 42-44 only, not split variability. Threshold for F1 chosen on training out-of-fold predictions. Test set was not used for any selection. Semer (near-constant, fictitious drug) excluded. Dataset is small (about 1.9k rows), so no subsampling was needed. Substance columns are located by normalised name across features and targets.

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
