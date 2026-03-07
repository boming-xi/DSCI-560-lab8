import json
import matplotlib.pyplot as plt


# read summary results from part 1 and part 2
with open("output/part1/part1_summary.json", "r") as f:
    part1_data = json.load(f)

with open("output/part2/part2_summary.json", "r") as f:
    part2_data = json.load(f)


# summaries are already sorted by silhouette score (descending)
best_part1 = part1_data[0]
best_part2 = part2_data[0]

print("Best Doc2Vec result:")
print(best_part1)

print("\nBest Word2Vec + Bag-of-Bins result:")
print(best_part2)


# collect silhouette scores for plotting
part1_names = [item["name"] for item in part1_data]
part1_silhouette = [item["silhouette_score"] for item in part1_data]

part2_names = [item["name"] for item in part2_data]
part2_silhouette = [item["silhouette_cosine"] for item in part2_data]


# plot comparison
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.barh(part1_names, part1_silhouette)
plt.title("Doc2Vec Silhouette Scores")
plt.xlabel("Score")

plt.subplot(1, 2, 2)
plt.barh(part2_names, part2_silhouette)
plt.title("Word2Vec Silhouette Scores")
plt.xlabel("Score")

plt.tight_layout()
plt.savefig("part3_comparison.png")
plt.show()


# simple comparison output
print("\nComparison based on silhouette score:")

print(
    "Best Doc2Vec:",
    best_part1["name"],
    "| Score =",
    best_part1["silhouette_score"]
)

print(
    "Best Word2Vec:",
    best_part2["name"],
    "| Score =",
    best_part2["silhouette_cosine"]
)

if best_part1["silhouette_score"] > best_part2["silhouette_cosine"]:
    print("\nDoc2Vec performs better on this dataset.")
else:
    print("\nWord2Vec + Bag-of-Bins performs better on this dataset.")