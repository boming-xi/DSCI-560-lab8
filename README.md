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

---

## Part 2: Word2Vec + Bag-of-Bins Document Embeddings

5. `part2_word2vec.py` trains a single Word2Vec model, clusters words into semantic bins, builds document vectors two ways (raw frequency and TF-IDF weighted), clusters the documents, and writes evaluation results.

**Run Part 2**
1. Install dependencies if not already installed.

```bash
python -m pip install gensim scikit-learn numpy
```

2. Run the script from this folder.

```bash
python part2_word2vec.py --input data/clean.json --output-dir output/part2
```

3. Optional arguments.

```
--k-min               minimum number of document clusters to search (default: 2)
--k-max               maximum number of document clusters to search (default: 10)
--samples-per-cluster number of sample documents shown per cluster (default: 3)
--text-field          field to use as document text (default: clean_text)
--seed                random seed for reproducibility (default: 42)
```

4. Check the outputs.

`output/part2/part2_summary.txt` and `output/part2/part2_results.csv` summarize all configurations.
Each config folder (e.g. `output/part2/w2v_k50_freq/`) contains:
- `cluster_report.json` and `cluster_report.txt` — per-cluster details and sample documents
- `cluster_assignments.csv` — document-to-cluster mapping
- `word_bins.json` — top representative words for each semantic bin

**Word bin configurations tested:** K = 3, 5, 8, 10, 20, 50, 100, 200 (both frequency and TF-IDF weighted for each).


## Part 3: Comparative Analysis

1. `part3_analysis.py` compares the clustering performance of Doc2Vec (Part 1) and Word2Vec + Bag-of-Bins (Part 2).  
   The script reads the summary outputs from both methods and generates a silhouette score comparison plot.

### Run Part 3

1. Make sure Part 1 and Part 2 have already been executed and the following files exist:

   - `output/part1/part1_summary.json`
   - `output/part2/part2_summary.json`

2. Run the script from this folder.

```bash
python part3_analysis.py

3.Optional arguments.
(No optional arguments required for Part 3)

4.Check the outputs.
- part3_comparison.png — silhouette score comparison between Doc2Vec and Word2Vec embeddings
- Terminal output — best configuration from each method and their silhouette scores
