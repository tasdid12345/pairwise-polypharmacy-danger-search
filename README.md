README.md unchanged. The method, dataset, baseline, split and metrics are the same; only the column lookup in load_data() was made robust (substance columns are matched by normalised name across the features and targets frames, instead of a direct features['Alcohol'] lookup that raised KeyError). The README was not provided to me, so it has not been edited.

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
