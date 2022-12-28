import csv
import os
"""
Serialize DeepMatcher datasets (tableA.csv and tableB.csv) into BERT-styled txt files like Ditto's
Input: ./data/**
Output: ./**
"""

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

output_paths = {
  'Structured/Amazon-Google': './Structured/Amazon-Google',
  'Structured/DBLP-ACM': './Structured/DBLP-ACM',
  'Structured/DBLP-GoogleScholar': './Structured/DBLP-GoogleScholar',
  'Structured/Beer': './Structured/Beer',
  'Structured/Walmart-Amazon': './Structured/Walmart-Amazon',

  'Dirty/DBLP-ACM': './Dirty/DBLP-ACM',
  'Dirty/Walmart-Amazon': './Dirty/Walmart-Amazon',

  'Textual/Abt-Buy': './Textual/Abt-Buy',
  'Textual/Amazon-Google': './Textual/Amazon-Google',
  'Textual/IMDB-TVDB': './Textual/IMDB-TVDB',
}

for ds in datasets:
  tableA_csv = datasets[ds] + '/tableA.csv'
  tableB_csv = datasets[ds] + '/tableB.csv'

  os.makedirs(os.path.dirname(output_paths[ds] + '/tableA.txt'), exist_ok=True)
  with open(output_paths[ds] + '/tableA.txt', 'w') as out_file:
    with open(tableA_csv, 'r') as in_file:
      reader = csv.reader(in_file)
      header = next(reader)
      header = header[1:] # skip ID attribute
      for row in reader:
        row = row[1:] # skip ID attribute

        new_row = row[0]
        if 'Textual' not in ds:
          new_row = ' '.join([f"COL {h} VAL {r}" for h, r in zip(header, row)])
        out_file.write(new_row + '\n')

    out_file.truncate(out_file.tell() - 1) # remove the last newline

  os.makedirs(os.path.dirname(output_paths[ds] + '/tableB.txt'), exist_ok=True)
  with open(output_paths[ds] + '/tableB.txt', 'w') as out_file:
    with open(tableB_csv, 'r') as in_file:
      reader = csv.reader(in_file)
      header = next(reader)
      header = header[1:] # skip ID attribute
      for row in reader:
        row = row[1:] # skip ID attribute

        new_row = row[0]
        if 'Textual' not in ds:
          new_row = ' '.join([f"COL {h} VAL {r}" for h, r in zip(header, row)])
        out_file.write(new_row + '\n')

    out_file.truncate(out_file.tell() - 1) # remove the last newline
