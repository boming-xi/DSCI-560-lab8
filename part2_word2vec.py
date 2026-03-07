from __future__ import annotations
import argparse
import csv
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class BinConfig:
    k: int
    @property
    def name(self) -> str:
        return f"w2v_k{self.k}"


# include small K for sensitivity plus required K
BIN_CONFIGS = [
    BinConfig(k=3),
    BinConfig(k=5),
    BinConfig(k=8),
    BinConfig(k=10),
    BinConfig(k=20),
    BinConfig(k=50),
    BinConfig(k=100),
    BinConfig(k=200),
]

# word2vec params
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_SG = 1
W2V_NEGATIVE = 10
W2V_SAMPLE = 1e-4


def load_posts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()
    if not content:
        return []
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_text(post: Dict, field: str) -> str:
    if field in post and post.get(field):
        return str(post[field])
    title = post.get("title") or ""
    body = post.get("selftext") or post.get("body") or ""
    return f"{title} {body}".strip()


def tokenize(texts: List[str]) -> List[List[str]]:
    return [simple_preprocess(t, deacc=True, min_len=2) for t in texts]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_word2vec(token_lists: List[List[str]], seed: int) -> Word2Vec:
    return Word2Vec(
        sentences=token_lists,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,
        negative=W2V_NEGATIVE,
        sample=W2V_SAMPLE,
        epochs=W2V_EPOCHS,
        seed=seed,
        workers=1,
    )


@dataclass
class WordBins:
    k: int
    word_to_bin: Dict[str, int]
    bin_top_words: Dict[int, List[str]]
    def bin_for(self, word: str) -> Optional[int]:
        return self.word_to_bin.get(word)


def cluster_words(model: Word2Vec, k: int, seed: int, top_n: int = 15) -> WordBins:
    vocab = list(model.wv.index_to_key)
    vectors = np.array([model.wv[w] for w in vocab])

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(vectors)

    word_to_bin = {w: int(labels[i]) for i, w in enumerate(vocab)}

    bin_top_words = {}
    for bin_id in range(k):
        idx = [i for i, lbl in enumerate(labels) if lbl == bin_id]
        if not idx:
            bin_top_words[bin_id] = []
            continue
        centroid = km.cluster_centers_[bin_id]
        bin_vecs = vectors[idx]
        sims = cosine_similarity(bin_vecs, centroid.reshape(1, -1)).ravel()
        ranked = sorted(zip(idx, sims), key=lambda x: x[1], reverse=True)
        bin_top_words[bin_id] = [vocab[i] for i, _ in ranked[:top_n]]
    return WordBins(k=k, word_to_bin=word_to_bin, bin_top_words=bin_top_words)


def build_freq_vectors(token_lists, bins):
    N, K = len(token_lists), bins.k
    matrix = np.zeros((N, K))
    for i, tokens in enumerate(token_lists):
        in_vocab = 0
        for tok in tokens:
            b = bins.bin_for(tok)
            if b is not None:
                matrix[i, b] += 1
                in_vocab += 1
        if in_vocab > 0:
            matrix[i] /= in_vocab
    return matrix


def build_tfidf_vectors(token_lists, bins):
    N, K = len(token_lists), bins.k
    tf = np.zeros((N, K))
    df = np.zeros(K)
    for i, tokens in enumerate(token_lists):
        in_vocab = 0
        seen_bins = set()
        for tok in tokens:
            b = bins.bin_for(tok)
            if b is not None:
                tf[i, b] += 1
                seen_bins.add(b)
                in_vocab += 1
        if in_vocab > 0:
            tf[i] /= in_vocab
        for b in seen_bins:
            df[b] += 1
    idf = np.log((N + 1) / (df + 1)) + 1
    matrix = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms
    return matrix


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def find_best_k(vectors, k_min, k_max, seed):
    best_k = k_min
    best_score = -1.0
    scores = {}
    max_k = min(k_max, max(2, vectors.shape[0] - 1))
    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(vectors, labels, metric="cosine"))
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, best_score, scores


def evaluate_clustering(vectors, labels):
    """Returns silhouette (cosine), davies-bouldin, calinski-harabasz."""
    if len(set(labels)) < 2:
        return 0.0, float("inf"), 0.0
    sil = float(silhouette_score(vectors, labels, metric="cosine"))
    db  = float(davies_bouldin_score(vectors, labels))
    ch  = float(calinski_harabasz_score(vectors, labels))
    return sil, db, ch


def cluster_balance_stats(labels: np.ndarray) -> Dict:
    """Compute cluster size distribution metrics."""
    counts = np.bincount(labels)
    min_size   = int(counts.min())
    max_size   = int(counts.max())
    size_ratio = round(float(max_size / min_size), 4) if min_size > 0 else float("inf")
    singletons = int(np.sum(counts == 1))
    return {
        "min_cluster_size":   min_size,
        "max_cluster_size":   max_size,
        "size_ratio":         size_ratio,
        "singleton_clusters": singletons,
    }


def build_cluster_report(
    vectors_norm, labels, texts, meta, bins,
    weighting, bin_config, silhouette, db_index, k_scores, args,
) -> Dict:
    k = int(labels.max()) + 1
    report: Dict = {
        "bin_config": bin_config.name,
        "k_word_bins": bins.k,
        "weighting": weighting,
        "doc_clusters_k": k,
        "num_docs": len(texts),
        "silhouette_cosine": round(silhouette, 4),
        "davies_bouldin": round(db_index, 4),
        "k_scores": {str(kk): round(v, 4) for kk, v in k_scores.items()},
        "clusters": [],
    }
    for cid in range(k):
        idx = [i for i, lbl in enumerate(labels) if lbl == cid]
        if not idx:
            continue
        cluster_vecs = vectors_norm[idx]
        centroid = cluster_vecs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_vecs, centroid).ravel()
        ranked = sorted(zip(idx, sims), key=lambda x: x[1], reverse=True)
        if len(idx) > 1:
            sim_mat = cosine_similarity(cluster_vecs)
            upper = sim_mat[np.triu_indices(len(idx), k=1)]
            avg_sim = float(np.mean(upper))
        else:
            avg_sim = 1.0
        samples = []
        for doc_idx, score in ranked[: args.samples_per_cluster]:
            post = meta[doc_idx]
            preview = textwrap.shorten(get_text(post, args.text_field), width=200, placeholder="...")
            samples.append({
                "id": post.get("id"),
                "title": post.get("title"),
                "preview": preview,
                "score": round(float(score), 4),
            })
        report["clusters"].append({
            "cluster": cid,
            "size": len(idx),
            "avg_intra_similarity": round(avg_sim, 4),
            "samples": samples,
        })
    return report


def write_cluster_txt(path: str, report: Dict, bins: WordBins) -> None:
    lines = [
        f"Config        : {report['bin_config']}",
        f"Word Bins K   : {report['k_word_bins']}",
        f"Weighting     : {report['weighting']}",
        f"Doc Clusters  : {report['doc_clusters_k']}",
        f"Documents     : {report['num_docs']}",
        f"Silhouette    : {report['silhouette_cosine']}",
        f"Davies-Bouldin: {report['davies_bouldin']}",
        "",
        "Word Bin Samples (top 5 bins shown):",
    ]
    for bin_id in range(min(5, bins.k)):
        top = bins.bin_top_words.get(bin_id, [])[:8]
        lines.append(f"  Bin {bin_id:>3}: {', '.join(top)}")
    lines.append("")
    for cl in report["clusters"]:
        lines.append(f"Cluster {cl['cluster']} (size={cl['size']}, avg_sim={cl['avg_intra_similarity']})")
        lines.append("Samples:")
        for s in cl["samples"]:
            lines.append(f"  - {s.get('id')}: {s.get('title')}")
            lines.append(f"    {s.get('preview')}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def save_assignments(path: str, meta: List[Dict], labels: np.ndarray, args: argparse.Namespace) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "title", "cluster", "clean_text"])
        for post, lbl in zip(meta, labels):
            writer.writerow([
                post.get("id"),
                post.get("title"),
                int(lbl),
                post.get(args.text_field) or get_text(post, args.text_field),
            ])


def write_results_csv(path: str, summary: List[Dict]) -> None:
    """Unified comparison CSV with all metrics — use this for Part 3 cross-method comparison."""
    fieldnames = [
        "name", "k_word_bins", "weighting", "doc_clusters_k",
        "silhouette_cosine", "davies_bouldin", "calinski_harabasz",
        "avg_intra_similarity",
        "min_cluster_size", "max_cluster_size", "size_ratio", "singleton_clusters",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def run_config(bin_config, w2v_model, token_lists, texts, meta, args):
    bins = cluster_words(w2v_model, bin_config.k, args.seed)
    summary_rows = []
    for weighting in ("freq", "tfidf"):
        if weighting == "freq":
            vectors = build_freq_vectors(token_lists, bins)
        else:
            vectors = build_tfidf_vectors(token_lists, bins)
        vectors_norm = l2_normalize(vectors)
        best_k, best_sil, k_scores = find_best_k(vectors_norm, args.k_min, args.k_max, args.seed)
        km = KMeans(n_clusters=best_k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(vectors_norm)

        sil, db, ch = evaluate_clustering(vectors_norm, labels)
        balance = cluster_balance_stats(labels)

        report = build_cluster_report(
            vectors_norm, labels, texts, meta, bins,
            weighting, bin_config, sil, db, k_scores, args,
        )
        weighted = sum(cl["avg_intra_similarity"] * cl["size"] for cl in report["clusters"])
        avg_intra = weighted / len(texts) if texts else 0.0

        out_dir = os.path.join(args.output_dir, f"{bin_config.name}_{weighting}")
        ensure_dir(out_dir)
        with open(os.path.join(out_dir, "cluster_report.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        write_cluster_txt(os.path.join(out_dir, "cluster_report.txt"), report, bins)
        save_assignments(os.path.join(out_dir, "cluster_assignments.csv"), meta, labels, args)
        word_bin_data = {str(bid): bins.bin_top_words.get(bid, []) for bid in range(bins.k)}
        with open(os.path.join(out_dir, "word_bins.json"), "w", encoding="utf-8") as fh:
            json.dump(word_bin_data, fh, ensure_ascii=False, indent=2)

        summary_rows.append({
            "name":                 f"{bin_config.name}_{weighting}",
            "k_word_bins":          bin_config.k,
            "weighting":            weighting,
            "doc_clusters_k":       best_k,
            "silhouette_cosine":    round(sil, 4),
            "davies_bouldin":       round(db, 4),
            "calinski_harabasz":    round(ch, 4),
            "avg_intra_similarity": round(avg_intra, 4),
            "min_cluster_size":     balance["min_cluster_size"],
            "max_cluster_size":     balance["max_cluster_size"],
            "size_ratio":           balance["size_ratio"],
            "singleton_clusters":   balance["singleton_clusters"],
        })
        print(
            f"  [{bin_config.name} | {weighting:5s}] "
            f"k={best_k} sil={sil:.4f} db={db:.4f} ch={ch:.1f} "
            f"ratio={balance['size_ratio']}"
        )
    return summary_rows


def parse_args():
    p = argparse.ArgumentParser(description="Lab 8 Part 2: Word2Vec + Bag-of-Bins document embeddings.")
    p.add_argument("--input",               default="data/clean.json")
    p.add_argument("--output-dir",          default="output/part2")
    p.add_argument("--text-field",          default="clean_text")
    p.add_argument("--k-min",               type=int, default=2)
    p.add_argument("--k-max",               type=int, default=10)
    p.add_argument("--samples-per-cluster", type=int, default=3)
    p.add_argument("--seed",                type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    posts = load_posts(args.input)
    if not posts:
        raise SystemExit(f"No posts found at: {args.input}")
    texts, meta = [], []
    for post in posts:
        t = get_text(post, args.text_field)
        if t:
            texts.append(t)
            meta.append(post)
    if len(texts) < 3:
        raise SystemExit("Not enough documents (need ≥ 3).")
    print(f"Loaded {len(texts)} documents.")
    token_lists = tokenize(texts)
    print("Training Word2Vec (skip-gram, negative sampling) …")
    w2v_model = train_word2vec(token_lists, args.seed)
    print(f"  Vocab size: {len(w2v_model.wv)}")
    ensure_dir(args.output_dir)

    all_summary: List[Dict] = []
    for bin_config in BIN_CONFIGS:
        print(f"\nBin config: {bin_config.name} (K={bin_config.k})")
        rows = run_config(bin_config, w2v_model, token_lists, texts, meta, args)
        all_summary.extend(rows)

    all_summary.sort(key=lambda r: (r["silhouette_cosine"], -r["davies_bouldin"]), reverse=True)

    with open(os.path.join(args.output_dir, "part2_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(all_summary, fh, ensure_ascii=False, indent=2)

    write_results_csv(os.path.join(args.output_dir, "part2_results.csv"), all_summary)

    header = (
        f"{'Config':<25} {'Bins':>5} {'Weight':>6} {'DocK':>5} "
        f"{'Sil':>8} {'DB':>8} {'CH':>10} {'AvgIntra':>9} "
        f"{'MinSz':>6} {'MaxSz':>6} {'Ratio':>7} {'Sing':>5}"
    )
    rows_txt = [
        "Lab 8 Part 2 — Word2Vec + Bag-of-Bins Summary",
        "=" * len(header), header, "-" * len(header),
    ]
    for r in all_summary:
        rows_txt.append(
            f"{r['name']:<25} {r['k_word_bins']:>5} {r['weighting']:>6} {r['doc_clusters_k']:>5} "
            f"{r['silhouette_cosine']:>8.4f} {r['davies_bouldin']:>8.4f} {r['calinski_harabasz']:>10.2f} "
            f"{r['avg_intra_similarity']:>9.4f} "
            f"{r['min_cluster_size']:>6} {r['max_cluster_size']:>6} {r['size_ratio']:>7.2f} "
            f"{r['singleton_clusters']:>5}"
        )
    rows_txt.append("-" * len(header))
    best = all_summary[0]
    rows_txt += [
        "",
        "Best configuration (silhouette ↑, DB ↓, CH ↑):",
        f"  {best['name']}  |  bins={best['k_word_bins']}  "
        f"|  sil={best['silhouette_cosine']:.4f}  "
        f"|  db={best['davies_bouldin']:.4f}  "
        f"|  ch={best['calinski_harabasz']:.2f}",
        "",
        "Metrics:",
        "  silhouette_cosine  — higher is better  [-1, 1]",
        "  davies_bouldin     — lower is better   [0, ∞)",
        "  calinski_harabasz  — higher is better  [0, ∞)",
        "  size_ratio         — max/min cluster size; closer to 1 = more balanced",
        "  singleton_clusters — clusters with only 1 document",
    ]
    with open(os.path.join(args.output_dir, "part2_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows_txt))

    print(f"\nDone. Results in: {args.output_dir}")
    print(f"Best config: {best['name']}  sil={best['silhouette_cosine']:.4f}  db={best['davies_bouldin']:.4f}  ch={best['calinski_harabasz']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())