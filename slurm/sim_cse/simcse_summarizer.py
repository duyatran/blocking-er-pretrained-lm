import os
import random
import numpy as np
import torch
import glob
import shutil

from summarizer import Summarizer

# SimCSE model implemented with sentence_transformers' MultipleNegativesRankingLoss, following example
# at https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/SimCSE/README.md
from sentence_transformers import SentencesDataset, SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

def train(hp):
    # Define sentence transformer model using CLS pooling to get one fixed-sized sentence vector
    model_names = {'distilbert': 'distilbert-base-uncased'}
    model_name = model_names[hp['lm']]

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=hp['device'])

    # Assume inputs from ditto_blocking/serializer.py: tableA.txt and tableB.txt
    tableA_fn = hp['dataset_path'] + '/tableA.txt'
    tableB_fn = hp['dataset_path'] + '/tableB.txt'
    train_sentences = []
    for line in open(tableA_fn):
      train_sentences.append(line)
    for line in open(tableB_fn):
      train_sentences.append(line)

    # TF-IDF summarizer
    summarizer = Summarizer(train_sentences, model_name)
    train_sentences = summarizer.transform_lines(train_sentences, max_len=32)
    examples = [InputExample(texts=[line, line]) for line in train_sentences]

    train_data = SentencesDataset(examples=examples, model=model)
    train_dataloader = DataLoader(train_data, batch_size=hp['batch_size'], shuffle=True)

    # Use the MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    if os.path.exists(hp['output_fn']):
        import shutil
        shutil.rmtree(hp['output_fn'])

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=hp['n_epochs'],
        show_progress_bar=True
    )

    model.save(hp['output_fn'])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

data_sets = {
  'Structured/Amazon-Google': '/u/dtran/slurm/input/Structured/Amazon-Google',
  'Structured/Beer': '/u/dtran/slurm/input/Structured/Beer',
  'Structured/DBLP-ACM': '/u/dtran/slurm/input/Structured/DBLP-ACM',
  'Structured/Walmart-Amazon': '/u/dtran/slurm/input/Structured/Walmart-Amazon',

  'Dirty/Walmart-Amazon': '/u/dtran/slurm/input/Dirty/Walmart-Amazon',
  'Dirty/DBLP-ACM': '/u/dtran/slurm/input/Dirty/DBLP-ACM',

  'Textual/Abt-Buy': '/u/dtran/slurm/input/Textual/Abt-Buy',
  'Textual/Amazon-Google': '/u/dtran/slurm/input/Textual/Amazon-Google',
  'Textual/IMDB-TVDB': '/u/dtran/slurm/input/Textual/IMDB-TVDB',
}

dataset_params = {
  'Structured/Amazon-Google': [(8, 5, 'cuda')],
  'Structured/Beer': [(8, 5, 'cuda')],
  'Structured/DBLP-ACM': [(8, 5, 'cuda')],
  'Structured/Walmart-Amazon': [(8, 5, 'cuda')],

  'Dirty/DBLP-ACM': [(8, 5, 'cuda')],
  'Dirty/Walmart-Amazon': [(8, 5, 'cuda')],

  'Textual/Abt-Buy': [(8, 5, 'cuda')],
  'Textual/Amazon-Google': [(8, 5, 'cuda')],
  'Textual/IMDB-TVDB': [(8, 5, 'cuda')],
}

lms = [
  'distilbert',
  # 'distilroberta',
  # 'albert'
  ]

seeds = [
  42,
  583,
  7714,
  34857,
  47359,
  ]

# values is a tuple of (model_fn, k, threshold)
blocking_params = {
  'Structured/Amazon-Google': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/Beer': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/DBLP-ACM': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/Walmart-Amazon': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],

  'Dirty/Walmart-Amazon': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Dirty/DBLP-ACM': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],

  'Textual/Amazon-Google': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Textual/IMDB-TVDB': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Textual/Abt-Buy': [('distilbert', k, [8]) for k in range(5, 200 + 1, 5)],
}

n_epochs_list = [
  5,
  # 10
  ]

for ds in data_sets:

  if ds not in dataset_params:
    continue

  for batch_size, n_epochs, device in dataset_params[ds]:
    for lm in lms:
      for seed in seeds:
        output_path = f"/u/dtran/slurm/sim_cse/output/{ds}/{seed}/{lm}/{n_epochs}/{batch_size}"

        # skip if model exists
        if os.path.isdir(output_path):
          continue

        hp = {
          'output_fn': output_path,
          'dataset_path': data_sets[ds],
          'batch_size': batch_size,
          'seed': seed,
          'device': device,
          'n_epochs': n_epochs,
          'lm': lm
        }

        seed_everything(hp['seed'])
        print(f"Training for {output_path}")
        train(hp)

        # Blocking
        model_paths = []
        for params in blocking_params[ds]:
          lm, k, train_batch_sizes = params
          model_dir = f'/u/dtran/slurm/sim_cse/output/{ds}'

          for train_batch_size in train_batch_sizes:
            model_fns = [(f'{model_dir}/{seed}/{lm}/{n_epochs}/{train_batch_size}', n_epochs) for n_epochs in n_epochs_list]
            model_paths += [model for model, _ in model_fns]

            for model_fn, n_epochs in model_fns:
              cmd = f"""python3 /u/dtran/slurm/sim_cse/blocker.py \
                --input_path {data_sets[ds]} \
                --left_fn tableA.txt \
                --right_fn tableB.txt \
                --model_fn {model_fn} \
                --seed {seed} \
                --n_epochs {n_epochs} \
                --lm {lm} \
                --k {k} \
                --train_batch_size {train_batch_size}"""

              print(cmd)
              os.system(cmd)

        # Clean up after blocking
        print('Removing temporary vector files after blocking')
        for temp_file in glob.glob(f'{data_sets[ds]}/*.simcse'):
          os.remove(temp_file)

        # Evaluate
        cmd = f"""python3 /u/dtran/slurm/sim_cse/evaluate.py --name {ds}"""

        print(cmd)
        os.system(cmd)

        # Remove model after evaluation
        shutil.rmtree(f'/u/dtran/slurm/sim_cse/output/{ds}')

