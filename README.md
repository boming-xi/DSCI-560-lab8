# DSCI 560 Lab 8

This folder contains the Lab 8 Part 1 implementation using the Lab 5 Reddit dataset and results.

**Contents**
1. `data/clean.json` and `data/raw.json` copied from Lab 5.
2. `lab5_results/cluster_output` copied from Lab 5 for reference.
3. `part1_doc2vec.py` to generate Doc2Vec embeddings, cluster with cosine distance, and output reports.
4. `output/` for Part 1 results.

**Run Part 1**
1. Install dependencies if needed.

```bash
python -m pip install gensim
```

2. Run the script from this folder.

```bash
python part1_doc2vec.py --input data/clean.json --output-dir output
```

3. Check the outputs.
`output/part1_report.txt` and `output/part1_summary.txt` summarize the comparison.
Each config folder contains `cluster_report.json`, `cluster_report.txt`, and `cluster_assignments.csv`.
