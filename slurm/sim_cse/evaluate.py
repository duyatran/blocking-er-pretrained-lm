import os
import time
import csv
import jsonlines
import glob
import argparse

def evaluate(experiments):
  # =============================================================================
  # Hardcoded values referenced and resolved from many sources (uleipzig, deepmatcher, Zenodo)
  # Tuples of (cross-join size, tableA-is-smaller flag)
  dataset_sizes = {
      'Structured/Amazon-Google': (1363 * 3226, True),
      'Structured/DBLP-ACM': (2616 * 2294, False),
      'Structured/Beer': (4345 * 3000, False),
      'Structured/Walmart-Amazon': (2554 * 22074, True),

      'Dirty/Walmart-Amazon': (2554 * 22074, True),
      'Dirty/DBLP-ACM': (2616 * 2294, False),

      'Textual/Abt-Buy': (1076 * 1076, True),
      'Textual/Amazon-Google': (1354 * 3039, True),
      'Textual/IMDB-TVDB': (5118 * 7810, True),
    }

  experiment_candidates = {
    'Structured/Amazon-Google': glob.glob('/u/dtran/slurm/sim_cse/output/Structured/Amazon-Google/**/candidates*.jsonl', recursive=True),
    'Structured/DBLP-ACM': glob.glob('/u/dtran/slurm/sim_cse/output/Structured/DBLP-ACM/**/candidates*.jsonl', recursive=True),
    'Structured/Beer': glob.glob('/u/dtran/slurm/sim_cse/output/Structured/Beer/**/candidates*.jsonl', recursive=True),
    'Structured/Walmart-Amazon': glob.glob('/u/dtran/slurm/sim_cse/output/Structured/Walmart-Amazon/**/candidates*.jsonl', recursive=True),

    'Dirty/Walmart-Amazon': glob.glob('/u/dtran/slurm/sim_cse/output/Dirty/Walmart-Amazon/**/candidates*.jsonl', recursive=True),
    'Dirty/DBLP-ACM': glob.glob('/u/dtran/slurm/sim_cse/output/Dirty/DBLP-ACM/**/candidates*.jsonl', recursive=True),

    'Textual/Abt-Buy': glob.glob('/u/dtran/slurm/sim_cse/output/Textual/Abt-Buy/**/candidates*.jsonl', recursive=True),
    'Textual/Amazon-Google': glob.glob('/u/dtran/slurm/sim_cse/output/Textual/Amazon-Google/**/candidates*.jsonl', recursive=True),
    'Textual/IMDB-TVDB': glob.glob('/u/dtran/slurm/sim_cse/output/Textual/IMDB-TVDB/**/candidates*.jsonl', recursive=True),
  }

  experiment_dict = {
    'Structured/Amazon-Google': '/u/dtran/slurm/data/Structured/Amazon-Google/matches_1300.csv',
    'Structured/Beer': '/u/dtran/slurm/data/Structured/Beer/matches.csv',
    'Structured/DBLP-ACM': '/u/dtran/slurm/data/Structured/DBLP-ACM/matches_2224.csv',
    'Structured/Walmart-Amazon': '/u/dtran/slurm/data/Structured/Walmart-Amazon/matches.csv',

    'Dirty/Walmart-Amazon': '/u/dtran/slurm/data/Dirty/Walmart-Amazon/matches.csv',
    'Dirty/DBLP-ACM': '/u/dtran/slurm/data/Dirty/DBLP-ACM/matches_2224.csv',

    'Textual/Abt-Buy': '/u/dtran/slurm/data/Textual/Abt-Buy/matches.csv',
    'Textual/Amazon-Google': '/u/dtran/slurm/data/Textual/Amazon-Google/matches.csv',
    'Textual/IMDB-TVDB': '/u/dtran/slurm/data/Textual/IMDB-TVDB/matches.csv',
  }

  # =============================================================================
  # Prepare the results file
  result_file_name = './results/eval.txt'
  res_file = open(result_file_name, 'a')
  res_file.write(os.linesep + os.linesep + 'Experiment %s:' % (\
                time.strftime('%Y%m%d-%H%M')) + os.linesep)
  # =============================================================================

  # =============================================================================
  # Run experiments

  for ds_name in experiments:
    print('='*70)
    print
    print('Data set:', ds_name, '      ',  time.ctime())
    print('----------' + '-' * len(ds_name))

    # read in golden matches
    match_pairs = None
    matches_csv = experiment_dict[ds_name]
    cross_join_size, tableA_smaller = dataset_sizes[ds_name]

    with open(matches_csv) as f:
      reader = csv.reader(f)
      next(reader) # skip header

      # blocker picks top-K from smaller table, and match files are A == B, so if table A is bigger, reverse it,
      match_pairs = set(tuple(map(int, reversed(line))) if tableA_smaller else tuple(map(int, line)) for line in reader)

    for candidate_fn in experiment_candidates[ds_name]:
      pairs = [] # list of candidate pairs (each as a tuple)
      with jsonlines.open(candidate_fn) as reader:
        for idx, row in enumerate(reader):
          pairs.append((int(row[0]), int(row[1])))


      # Evaluate metrics
      m =  0  # Number of matches
      nm = 0  # Number of non-matches

      for pair in pairs:
        matched = pair in match_pairs
        if matched:
          print
        if (matched == True):
          m += 1
        else:
          nm += 1

      num_rec_pairs = len(pairs)
      assert (m+nm) == len(pairs)

      tm = len(match_pairs)
      print('** Total number of true matches:', len(match_pairs))

      if (tm > 0):
        assert tm >= m
        pc = float(m) / float(tm)
      else:
        pc = -1

      print('    Pairs completeness:  %.2f %%' % (pc*100.0))

      pq = float(m) / float(num_rec_pairs)
      print('    Pairs quality:       %.2f %%' % (pq*100.0))

      fscore = (pc*pq)/(pc+pq)
      print('    F-score:             %.2f %%' % (fscore*100.0))

      cssr = float(num_rec_pairs) / (cross_join_size)
      rr = 1 - cssr
      print(f'    |C|:                 {num_rec_pairs}')
      print('    CSSR:                %.2f' % (cssr))
      print('    Reduction ratio:     %.2f %%' % (rr*100.0))

      res_file_str = '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d' % \
                    (candidate_fn, pc, pq, fscore, rr, cssr, num_rec_pairs)
      print(f'    Saved into results file: {res_file_str}')
      res_file.write(res_file_str + os.linesep)

      print
      print
  # =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    params = parser.parse_args()

    evaluate([params.name])
