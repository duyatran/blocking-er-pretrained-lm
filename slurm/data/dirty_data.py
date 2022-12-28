import csv
import random

file_names = ['tableA.csv', 'tableB.csv']

for fn in file_names:
  with open(f"./Structured/Amazon-Google/{fn}") as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)

  for row in csv_data:
    for idx in range(2, 4):
      rand_num = random.uniform(0, 1)

      if rand_num < 0.5:
        row[1] = f'{row[1]} {row[idx]}'
        row[idx] = ''

  with open(f"./Dirty/Amazon-Google/{fn}", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    for row in csv_data:
      csv_writer.writerow(row)