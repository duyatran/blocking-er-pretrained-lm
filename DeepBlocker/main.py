import os
import time
import gc
import pandas as pd
from pathlib import Path
import fasttext

from deep_blocker import DeepBlocker
from tuple_embedding_models import AutoEncoderTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
from configurations import *

datasets = {
    'Structured/Amazon-Google': '../data/Structured/Amazon-Google',
    'Structured/DBLP-ACM': '../data/Structured/DBLP-ACM',
    'Structured/DBLP-GoogleScholar': '../data/Structured/DBLP-GoogleScholar',
    'Structured/Beer': '../data/Structured/Beer',
    'Structured/Walmart-Amazon': '../data/Structured/Walmart-Amazon',

    'Dirty/DBLP-ACM': '../data/Dirty/DBLP-ACM',
    'Dirty/Walmart-Amazon': '../data/Dirty/Walmart-Amazon',

    'Textual/Abt-Buy': '../data/Textual/Abt-Buy',
    'Textual/Amazon-Google': '../data/Textual/Amazon-Google',
    'Textual/IMDB-TVDB': '../data/Textual/IMDB-TVDB',
}

dataset_params = {
    'Structured/Amazon-Google': (["title", "manufacturer", "price"], 'matches_1300.csv'),
    'Structured/DBLP-ACM': (['title', 'authors' , 'venue' , 'year'], 'matches_2224.csv'),
    'Structured/Beer': (['Beer_Name', 'Brew_Factory_Name', 'Style', 'ABV'], 'matches.csv'),
    'Structured/Walmart-Amazon': (['title', 'category', 'brand', 'modelno', 'price'], 'matches.csv'),

    'Dirty/DBLP-ACM': (['title', 'authors' , 'venue' , 'year'], 'matches_2224.csv'),
    'Dirty/Walmart-Amazon': (['title', 'category', 'brand', 'modelno', 'price'], 'matches.csv'),

    'Textual/Abt-Buy': (["aggregate value"], 'matches.csv'),
    'Textual/Amazon-Google': (["aggregate value"], 'matches.csv'),
    'Textual/IMDB-TVDB': (["aggregate value"], 'matches.csv'),
}

def build_blocks(vector_pairing_model, left_tuple_embeddings, right_tuple_embeddings):
    print("Indexing the embeddings from the right dataset")
    vector_pairing_model.index(right_tuple_embeddings)

    print("Querying the embeddings from left dataset")
    topK_neighbors = vector_pairing_model.query(left_tuple_embeddings)

    candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

    return candidate_set_df

def prepare_dfs(folder_root, match_fn):
    folder_root = Path(folder_root)
    # blocking_utils.process_files(str(folder_root))
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)
    golden_df = pd.read_csv(Path(folder_root) / match_fn)
    return left_df, right_df, golden_df

if __name__ == "__main__":

    #
    word_embedding_model = fasttext.load_model(FASTTEXT_EMBEDDIG_PATH)

    for ds in datasets:
        folder_root = datasets[ds]
        cols_to_block, match_fn = dataset_params[ds]
        left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"

        # File name for writing results to (results will be appended to this file)
        #
        result_file_name = f'./results/eval-{ds.replace("/", "-")}.res'

        # =============================================================================
        # Open the results file
        #
        res_file = open(result_file_name, 'a')
        res_file.write(os.linesep+os.linesep+'Experiment started %s:' % (\
                    time.strftime('%Y%m%d-%H%M')) + os.linesep)

        tuple_embedding_model = AutoEncoderTupleEmbedding(word_embedding_model)

        left_df, right_df, golden_df = prepare_dfs(folder_root, match_fn)

        db = DeepBlocker(tuple_embedding_model)
        left_tuple_embeddings, right_tuple_embeddings = db.create_embeddings(left_df, right_df, cols_to_block)

        for k in range(5, 200+1, 5):
            print(k)
            topK_vector_pairing_model = ExactTopKVectorPairing(K=k)
            candidate_set_df = build_blocks(topK_vector_pairing_model, left_tuple_embeddings, right_tuple_embeddings)
            statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)

            experiment_name = f'{ds}/k={k}'
            print(statistics_dict)
            res_file_str = '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d' % \
                (experiment_name,
                statistics_dict['recall'],
                statistics_dict['precision'],
                statistics_dict['fscore'],
                statistics_dict['rr'],
                statistics_dict['cssr'],
                statistics_dict['candidate_set_size'])
            res_file.write(res_file_str + os.linesep)

        gc.collect()

