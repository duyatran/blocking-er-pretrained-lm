## Source
- All code in this directory are from https://github.com/qcri/DeepBlocker
- Some minor modifications were made to facilitate running experiments continuously (embedding/model caching, garbage collection)

## Steps to run experiments
- Run `pip install -r requirements.txt` to install DeepBlocker dependencies
- Download and place fastText's Wiki word vectors (`wiki.en.bin` from [here](https://fasttext.cc/docs/en/pretrained-vectors.html)) in a new directory "embedding"
- Create a `results` directory
- Run `main.py` with at least 16 GB of available RAM (running on CPU is fine, there is an unresolved issue when running on GPU)
