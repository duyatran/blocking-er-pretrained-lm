import os
import random
import numpy as np
import torch
import nltk
import glob
import shutil

from summarizer import Summarizer

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, datasets, losses
from torch.utils.data import DataLoader



# TSDAE (Tranformer-based Denoising AutoEncoder) from example at
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/TSDAE/README.md
def train(hp):
    model_names = {'bert': 'bert-base-uncased'}
    model_name = model_names[hp['lm']]

    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Read in data and summarize
    tableA_fn = hp['dataset_path'] + '/tableA.txt'
    tableB_fn = hp['dataset_path'] + '/tableB.txt'
    train_sentences = []
    for line in open(tableA_fn):
      train_sentences.append(line)
    for line in open(tableB_fn):
      train_sentences.append(line)

    # TF-IDF summarizer
    summarizer = Summarizer(train_sentences, model_name)
    train_sentences = summarizer.transform_lines(train_sentences, max_len=16)
    print(train_sentences[0])

    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 2e-5},
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
  'bert',
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
  'Structured/Amazon-Google': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/Beer': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/DBLP-ACM': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Structured/Walmart-Amazon': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],

  'Dirty/Walmart-Amazon': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Dirty/DBLP-ACM': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],

  'Textual/Amazon-Google': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Textual/IMDB-TVDB': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
  'Textual/Abt-Buy': [('bert', k, [8]) for k in range(5, 200 + 1, 5)],
}

n_epochs_list = [
  5,
  # 10
  ]

# Only need to be run once, or simply download punkt in other ways
# nltk.download('punkt')
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





