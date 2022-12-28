## Pretrained Language Models for Blocking in Entity Resolution

### Reproduce DeepBlocker results
- Steps to reproduce results for DeepBlocker are in README.md in the DeepBlocker directory.

### Reproduce SimCSE/TSDAE results
- Install `sentence-transformers` v2.2.2: `pip install sentence-transformers==2.2.2`
- Original experiments were run on UofT slurm cluster by interactively running `tsdae_all_ds_train.py` and `simcse_all_ds_train.py`. The summarizer counterparts are also available.
- `slurm/data` and `slurm/input` are identical (duplicated for convenience when copying to cluster)
- `charts` contains raw results (`experiments-distilbert.xlsx`) and chart plotting notebook.
