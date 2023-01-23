## Pretrained Language Models for Blocking in Entity Resolution (ER)
### Introduction
- Achieved state-of-the-art results for ER's blocking step for a deep-learning-based method
- Project for UofT's [CSC2508 - Advanced Data Management System](https://koudas.github.io/csc2508-fall2022.html)
- See [technical report](https://github.com/duyatran/blocking-proj/blob/master/%5BCSC2508%5D%20Duy%20Tran%20-%20Project%20Report.pdf) for more details

### Reproduce DeepBlocker results
- Steps to reproduce results for DeepBlocker are in README.md in the DeepBlocker directory.

### Reproduce the report's SimCSE/TSDAE results
- Install `sentence-transformers` v2.2.2: `pip install sentence-transformers==2.2.2`
- Original experiments were run on UofT slurm cluster by interactively running `tsdae_all_ds_train.py` and `simcse_all_ds_train.py`. The summarizer counterparts are also available.
- `slurm/data` and `slurm/input` are identical (duplicated for convenience when copying to cluster)
- `charts` contains raw results (`experiments-distilbert.xlsx`) and chart plotting notebook.
