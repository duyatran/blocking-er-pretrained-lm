# Top-K cosine search implemented by Ditto

import os
import sys
import jsonlines
import pickle
import numpy as np
import argparse

from sentence_transformers import SentenceTransformer

def encode_all(path, input_fn, train_batch_size, seed, n_epochs, lm, model, overwrite=False):
    """Encode a collection of entries and output to a file

    Args:
        path (str): the input path
        input_fn (str): the file of the serialzied entries
        model (SentenceTransformer): the transformer model
        overwrite (boolean, optional): whether to overwrite out_fn

    Returns:
        List of str: the serialized entries
        List of np.ndarray: the encoded vectors
    """
    input_fn = os.path.join(path, input_fn)
    output_fn = f'{input_fn}.{seed}.{lm}.{n_epochs}.{train_batch_size}.simcse'

    # read from input_fn
    lines = open(input_fn).read().split('\n')

    # encode and dump
    if not os.path.exists(output_fn) or overwrite:
        vectors = model.encode(lines)
        vectors = [v / np.linalg.norm(v) for v in vectors]
        pickle.dump(vectors, open(output_fn, 'wb'))
    else:
        vectors = pickle.load(open(output_fn, 'rb'))
    return vectors


def blocked_matmul(mata, matb,
                   k=None,
                   batch_size=512):
    """Find the most similar pairs of vectors from two matrices (top-k)

    Args:
        mata (np.array): the first matrix
        matb (np.array): the second matrix
        k (int, optional): if set, return for each row in matb the top-k
            most similar vectors in mata
        batch_size (int, optional): the batch size of each block

    Returns:
        list of tuples: the pairs of similar vectors' indices and the similarity
    """
    results = []
    for start in range(0, len(matb), batch_size):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())
        if k is not None:
            indices = np.argpartition(-sim_mat, k, axis=0)
            for row in indices[:k]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    print(len(results))
    return results


def dump_pairs(out_fn, pairs):
    """Dump the pairs to a jsonl file
    """
    print(out_fn)
    with jsonlines.open(out_fn, mode='w') as writer:
        for idx_a, idx_b, score in pairs:
            writer.write([int(idx_a), int(idx_b)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--left_fn", type=str, default=None)
    parser.add_argument("--right_fn", type=str, default=None)
    parser.add_argument("--model_fn", type=str, required=True)
    parser.add_argument("--lm", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--k", type=int, default=None)
    hp = parser.parse_args()

    candidate_method = f'topk-{hp.k}' if hp.k is not None else f'threshold-{hp.threshold}'
    output_fn = f"{hp.model_fn}/candidates-{candidate_method}.jsonl"

    if not os.path.exists(output_fn):
        # load the model
        model = SentenceTransformer(hp.model_fn)

        # generate the vectors
        mata = matb = None
        # entries_a = entries_b = None
        if hp.left_fn is not None:
            mata = encode_all(hp.input_path, hp.left_fn, hp.train_batch_size, hp.seed, hp.n_epochs, hp.lm, model)
        if hp.right_fn is not None:
            matb = encode_all(hp.input_path, hp.right_fn, hp.train_batch_size, hp.seed, hp.n_epochs, hp.lm, model)

        if mata and matb:
            mata = np.array(mata)
            matb = np.array(matb)

            # for each item in smaller_mat, pick top-K
            smaller_mat, bigger_mat = (mata, matb) if mata.shape[0] <= matb.shape[0] else (matb, mata)

            pairs = blocked_matmul(bigger_mat, smaller_mat,
                    k=hp.k,
                    batch_size=hp.batch_size)
            dump_pairs(output_fn, pairs)
