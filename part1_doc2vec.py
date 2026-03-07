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
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
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


def parse_k_values(raw: str) -> List[int]:
    if not raw:
        return []
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"Invalid K value: {part}") from exc
    return sorted(set(values))


def resolve_k_values(args: argparse.Namespace, num_docs: int) -> Tuple[List[int], List[int]]:
    if args.k_values:
        requested = parse_k_values(args.k_values)
    else:
        requested = list(range(args.k_min, args.k_max + 1))
    max_k = max(2, num_docs - 1)
    valid = [k for k in requested if 2 <= k <= max_k]
    skipped = [k for k in requested if k not in valid]
    return valid, skipped


def compute_avg_intra_similarity(vectors: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    total_docs = len(vectors)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) <= 1:
            total += 1.0 * len(idx)
            continue
        sub = vectors[idx]
        sim_mat = cosine_similarity(sub)
        upper = sim_mat[np.triu_indices(len(idx), k=1)]
        avg = float(np.mean(upper)) if upper.size > 0 else 1.0
        total += avg * len(idx)
    return float(total / total_docs) if total_docs else 0.0


def evaluate_k_sweep(
    config: Doc2VecConfig,
    vectors_norm: np.ndarray,
    k_values: List[int],
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(vectors_norm)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(vectors_norm, labels, metric="cosine"))
        db = float(davies_bouldin_score(vectors_norm, labels))
        ch = float(calinski_harabasz_score(vectors_norm, labels))
        avg_intra = compute_avg_intra_similarity(vectors_norm, labels)
        sizes = [int(np.sum(labels == i)) for i in range(k)]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        ratio = (max_size / min_size) if min_size else float("inf")
        singletons = sum(1 for s in sizes if s == 1)
        rows.append(
            {
                "config": config.name,
                "vector_size": config.vector_size,
                "k": k,
                "silhouette_cosine": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 2),
                "avg_intra_similarity": round(avg_intra, 4),
                "min_cluster_size": min_size,
                "max_cluster_size": max_size,
                "size_ratio": round(ratio, 2) if ratio != float("inf") else ratio,
                "singleton_clusters": singletons,
            }
        )
    return rows


def write_k_sweep_outputs(
    out_dir: str,
    rows: List[Dict],
    k_values: List[int],
    skipped: List[int],
) -> None:
    if not rows:
        return
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "k_sweep_summary.csv")
    header = [
        "config",
        "vector_size",
        "k",
        "silhouette_cosine",
        "davies_bouldin",
        "calinski_harabasz",
        "avg_intra_similarity",
        "min_cluster_size",
        "max_cluster_size",
        "size_ratio",
        "singleton_clusters",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col) for col in header])

    lines = [
        "Doc2Vec K Sweep Summary",
        f"K values: {', '.join(str(k) for k in k_values)}",
    ]
    if skipped:
        lines.append(f"Skipped (invalid for n_docs): {', '.join(str(k) for k in skipped)}")
    lines += [
        "",
        f"{'Config':<10} {'Vec':>4} {'K':>3} {'Sil':>7} {'DB':>7} {'CH':>9} "
        f"{'AvgIntra':>9} {'Min':>4} {'Max':>4} {'Ratio':>6} {'Singles':>7}",
        "-" * 90,
    ]
    for row in rows:
        ratio = row["size_ratio"]
        ratio_str = "inf" if ratio == float("inf") else f"{ratio:>6.2f}"
        lines.append(
            f"{row['config']:<10} {row['vector_size']:>4} {row['k']:>3} "
            f"{row['silhouette_cosine']:>7.4f} {row['davies_bouldin']:>7.4f} "
            f"{row['calinski_harabasz']:>9.2f} {row['avg_intra_similarity']:>9.4f} "
            f"{row['min_cluster_size']:>4} {row['max_cluster_size']:>4} "
            f"{ratio_str:>6} {row['singleton_clusters']:>7}"
        )
    txt_path = os.path.join(out_dir, "k_sweep_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def find_best_k(
    vectors: np.ndarray,
    k_values: List[int],
    seed: int,
) -> Tuple[int, float, Dict[int, float]]:
    best_k = k_values[0]
    best_score = -1.0
    scores: Dict[int, float] = {}

    for k in k_values:
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


def cluster_and_report(
    config: Doc2VecConfig,
    texts: List[str],
    meta: List[Dict],
    vectors_norm: np.ndarray,
    X_tfidf,
    feature_names: np.ndarray,
    k: int,
    args: argparse.Namespace,
    out_dir: str,
    seed: int,
    k_scores: Dict[int, float] | None = None,
    silhouette_override: float | None = None,
) -> Dict:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(vectors_norm)

    silhouette = (
        float(silhouette_override)
        if silhouette_override is not None
        else float(silhouette_score(vectors_norm, labels, metric="cosine"))
    )

    report = {
        "config": asdict(config),
        "k": int(k),
        "num_docs": len(texts),
        "silhouette_score": round(float(silhouette), 4),
        "k_scores": {str(kk): round(float(vv), 4) for kk, vv in (k_scores or {}).items()},
        "clusters": [],
    }

    for cluster_id in range(k):
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
                    int(label),
                    post.get(args.text_field) or get_text(post, args.text_field),
                ]
            )

    return report


def build_reports(
    config: Doc2VecConfig,
    texts: List[str],
    meta: List[Dict],
    vectors_norm: np.ndarray,
    X_tfidf,
    feature_names: np.ndarray,
    k_values: List[int],
    args: argparse.Namespace,
    out_dir: str,
    seed: int,
) -> Dict:
    best_k, best_score, k_scores = find_best_k(vectors_norm, k_values, seed)
    return cluster_and_report(
        config=config,
        texts=texts,
        meta=meta,
        vectors_norm=vectors_norm,
        X_tfidf=X_tfidf,
        feature_names=feature_names,
        k=best_k,
        args=args,
        out_dir=out_dir,
        seed=seed,
        k_scores=k_scores,
        silhouette_override=best_score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 8 Part 1: Doc2Vec embeddings + cosine clustering.")
    parser.add_argument("--input", default="data/clean.json", help="Path to cleaned JSON data.")
    parser.add_argument("--output-dir", default="output", help="Directory to write results.")
    parser.add_argument("--text-field", default="clean_text")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument(
        "--k-values",
        default="",
        help="Comma-separated list of K values (overrides k-min/k-max).",
    )
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="Evaluate all K in range and write k_sweep_summary outputs.",
    )
    parser.add_argument(
        "--emit-k-reports",
        action="store_true",
        help="Write cluster_report/assignments for each K into per-K subfolders.",
    )
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

    k_values, skipped_k = resolve_k_values(args, len(texts))
    if not k_values:
        raise SystemExit("No valid K values after filtering. Adjust --k-values or k-min/k-max.")

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    token_lists = tokenize(texts)
    tagged_docs = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(token_lists)]

    summary = []
    k_sweep_rows: List[Dict] = []
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
        vectors_norm = l2_normalize(vectors)
        out_dir = os.path.join(args.output_dir, config.name)
        report = build_reports(
            config=config,
            texts=texts,
            meta=meta,
            vectors_norm=vectors_norm,
            X_tfidf=X_tfidf,
            feature_names=feature_names,
            k_values=k_values,
            args=args,
            out_dir=out_dir,
            seed=args.seed,
        )
        if args.emit_k_reports:
            for k in k_values:
                k_dir = os.path.join(out_dir, f"k{k}")
                cluster_and_report(
                    config=config,
                    texts=texts,
                    meta=meta,
                    vectors_norm=vectors_norm,
                    X_tfidf=X_tfidf,
                    feature_names=feature_names,
                    k=k,
                    args=args,
                    out_dir=k_dir,
                    seed=args.seed,
                )
        if args.k_sweep:
            sweep_rows = evaluate_k_sweep(config, vectors_norm, k_values, args.seed)
            k_sweep_rows.extend(sweep_rows)
            write_k_sweep_outputs(out_dir, sweep_rows, k_values, skipped_k)

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

    if args.k_sweep and k_sweep_rows:
        write_k_sweep_outputs(args.output_dir, k_sweep_rows, k_values, skipped_k)

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
        f"- K candidates: {', '.join(str(k) for k in k_values)}.",
        "- K selected by silhouette score (cosine) from the candidates above.",
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
