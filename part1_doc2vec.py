from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.utils import simple_preprocess
except Exception as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "Missing dependency: gensim. Install with `python -m pip install gensim` "
        "and re-run this script."
    ) from exc
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Doc2VecConfig:
    name: str
    vector_size: int
    min_count: int
    epochs: int
    window: int = 8
    dm: int = 1


DOC2VEC_CONFIGS: List[Doc2VecConfig] = [
    Doc2VecConfig(name="d2v_vs50", vector_size=50, min_count=2, epochs=40),
    Doc2VecConfig(name="d2v_vs100", vector_size=100, min_count=2, epochs=40),
    Doc2VecConfig(name="d2v_vs200", vector_size=200, min_count=2, epochs=40),
]


def load_posts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_text(post: Dict, text_field: str) -> str:
    if text_field in post and post.get(text_field):
        return str(post.get(text_field))
    title = post.get("title") or ""
    body = post.get("selftext") or post.get("body") or post.get("clean_text") or ""
    return f"{title} {body}".strip()


def tokenize(texts: Iterable[str]) -> List[List[str]]:
    return [simple_preprocess(text, deacc=True, min_len=2) for text in texts]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def find_best_k(
    vectors: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
) -> Tuple[int, float, Dict[int, float]]:
    best_k = k_min
    best_score = -1.0
    scores: Dict[int, float] = {}

    max_possible = min(k_max, max(2, vectors.shape[0] - 1))
    for k in range(k_min, max_possible + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(vectors, labels, metric="cosine"))
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_report_txt(path: str, report: Dict) -> None:
    lines: List[str] = []
    config = report.get("config", {})
    lines.append("Doc2Vec Config")
    lines.append(f"Name = {config.get('name')}")
    lines.append(f"Vector Size = {config.get('vector_size')}")
    lines.append(f"Min Count = {config.get('min_count')}")
    lines.append(f"Epochs = {config.get('epochs')}")
    lines.append(f"Window = {config.get('window')}")
    lines.append(f"DM = {config.get('dm')}")
    lines.append("")
    lines.append(f"K = {report['k']}")
    lines.append(f"Documents = {report['num_docs']}")
    lines.append(f"Silhouette Score (cosine) = {report.get('silhouette_score', 'N/A')}")
    lines.append("")

    for cluster in report["clusters"]:
        lines.append(f"Cluster {cluster['cluster']} (size={cluster['size']})")
        lines.append(f"Keywords: {', '.join(cluster['keywords'])}")
        lines.append(f"Avg Intra Similarity: {cluster['avg_intra_similarity']}")
        lines.append("Samples:")
        for sample in cluster["samples"]:
            lines.append(f"- {sample.get('id')}: {sample.get('title')}")
            lines.append(f"  {sample.get('preview')}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_reports(
    config: Doc2VecConfig,
    texts: List[str],
    meta: List[Dict],
    vectors: np.ndarray,
    args: argparse.Namespace,
    out_dir: str,
    seed: int,
) -> Dict:
    vectors_norm = l2_normalize(vectors)
    best_k, best_score, k_scores = find_best_k(vectors_norm, args.k_min, args.k_max, seed)

    kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(vectors_norm)

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    report = {
        "config": asdict(config),
        "k": int(best_k),
        "num_docs": len(texts),
        "silhouette_score": round(float(best_score), 4),
        "k_scores": {str(k): round(float(v), 4) for k, v in k_scores.items()},
        "clusters": [],
    }

    for cluster_id in range(best_k):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices:
            continue

        centroid_tfidf = np.mean(X_tfidf[indices].toarray(), axis=0)
        top_idx = centroid_tfidf.argsort()[::-1][: args.top_terms]
        keywords = [feature_names[i] for i in top_idx]

        cluster_vectors = vectors_norm[indices]
        centroid_vec = cluster_vectors.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_vectors, centroid_vec).ravel()
        ranked = sorted(zip(indices, sims), key=lambda x: x[1], reverse=True)

        if len(indices) > 1:
            sim_matrix = cosine_similarity(cluster_vectors)
            upper_triangle = sim_matrix[np.triu_indices(len(indices), k=1)]
            avg_similarity = float(np.mean(upper_triangle))
        else:
            avg_similarity = 1.0

        samples = []
        for doc_idx, score in ranked[: args.samples_per_cluster]:
            post = meta[doc_idx]
            preview = textwrap.shorten(
                get_text(post, args.text_field),
                width=200,
                placeholder="...",
            )
            samples.append(
                {
                    "id": post.get("id"),
                    "title": post.get("title"),
                    "preview": preview,
                    "score": round(float(score), 4),
                }
            )

        report["clusters"].append(
            {
                "cluster": cluster_id,
                "size": len(indices),
                "avg_intra_similarity": round(avg_similarity, 4),
                "keywords": keywords,
                "samples": samples,
            }
        )

    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "cluster_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    write_report_txt(os.path.join(out_dir, "cluster_report.txt"), report)

    csv_path = os.path.join(out_dir, "cluster_assignments.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "title", "cluster", "clean_text"])
        for post, label in zip(meta, labels):
            writer.writerow(
                [
                    post.get("id"),
                    post.get("title"),
                    label,
                    post.get(args.text_field) or get_text(post, args.text_field),
                ]
            )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 8 Part 1: Doc2Vec embeddings + cosine clustering.")
    parser.add_argument("--input", default="data/clean.json", help="Path to cleaned JSON data.")
    parser.add_argument("--output-dir", default="output", help="Directory to write results.")
    parser.add_argument("--text-field", default="clean_text")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    posts = load_posts(args.input)
    if not posts:
        raise SystemExit("No posts found.")

    texts = []
    meta = []
    for post in posts:
        text = get_text(post, args.text_field)
        if text:
            texts.append(text)
            meta.append(post)

    if len(texts) < 3:
        raise SystemExit("Not enough documents.")

    token_lists = tokenize(texts)
    tagged_docs = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(token_lists)]

    summary = []
    for config in DOC2VEC_CONFIGS:
        model = Doc2Vec(
            vector_size=config.vector_size,
            min_count=config.min_count,
            epochs=config.epochs,
            window=config.window,
            dm=config.dm,
            seed=args.seed,
            workers=1,
        )
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        vectors = np.vstack([model.dv[str(i)] for i in range(len(tagged_docs))])
        out_dir = os.path.join(args.output_dir, config.name)
        report = build_reports(config, texts, meta, vectors, args, out_dir, args.seed)

        weighted_intra = 0.0
        total_docs = 0
        for cluster in report["clusters"]:
            weighted_intra += cluster["avg_intra_similarity"] * cluster["size"]
            total_docs += cluster["size"]
        avg_intra = weighted_intra / total_docs if total_docs else 0.0

        summary.append(
            {
                "name": config.name,
                "vector_size": config.vector_size,
                "min_count": config.min_count,
                "epochs": config.epochs,
                "k": report["k"],
                "silhouette_score": report["silhouette_score"],
                "avg_intra_similarity": round(avg_intra, 4),
            }
        )

    summary_sorted = sorted(
        summary,
        key=lambda item: (item["silhouette_score"], item["avg_intra_similarity"]),
        reverse=True,
    )

    summary_path = os.path.join(args.output_dir, "part1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_sorted, handle, ensure_ascii=False, indent=2)

    best = summary_sorted[0]
    summary_lines = ["Doc2Vec Config Comparison (higher is better)"]
    summary_lines.append("")
    for item in summary_sorted:
        summary_lines.append(
            f"{item['name']} | vector_size={item['vector_size']} | k={item['k']} | "
            f"silhouette={item['silhouette_score']:.4f} | avg_intra={item['avg_intra_similarity']:.4f}"
        )
    summary_lines.append("")
    summary_lines.append("Best Config (quantitative):")
    summary_lines.append(
        f"{best['name']} with vector_size={best['vector_size']} "
        f"(silhouette={best['silhouette_score']:.4f}, avg_intra={best['avg_intra_similarity']:.4f})"
    )

    summary_txt_path = os.path.join(args.output_dir, "part1_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    report_path = os.path.join(args.output_dir, "part1_report.txt")
    report_lines = [
        "Lab 8 Part 1 Report",
        "",
        "Data Source:",
        f"- Input: {args.input}",
        f"- Documents: {len(texts)}",
        "",
        "Doc2Vec Configurations:",
        "- d2v_vs50: vector_size=50, min_count=2, epochs=40, window=8, dm=1",
        "- d2v_vs100: vector_size=100, min_count=2, epochs=40, window=8, dm=1",
        "- d2v_vs200: vector_size=200, min_count=2, epochs=40, window=8, dm=1",
        "",
        "Clustering:",
        "- KMeans on L2-normalized vectors (equivalent to cosine distance).",
        "- K selected by silhouette score (cosine) within the configured range.",
        "",
        "Quantitative Comparison:",
        *summary_lines[2:],
        "",
        "Conclusion:",
        f"- Best quantitative config: {best['name']} (vector_size={best['vector_size']}).",
        "- Review the cluster reports for qualitative coherence and note representative topics.",
    ]

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
