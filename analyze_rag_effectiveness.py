import json
import matplotlib.pyplot as plt

log_file = "rag_log.json"

with open(log_file, "r") as f:
    data = json.load(f)

mean_distances = []
verdict_labels = []

for entry in data:
    if "mean_distance" in entry and "verdict" in entry:
        mean_distances.append(entry["mean_distance"])
        verdict = entry["verdict"]
        if "B" in verdict:
            verdict_labels.append("RAG wins")
        else:
            verdict_labels.append("No RAG")

colors = ["green" if label == "RAG wins" else "red" for label in verdict_labels]

plt.figure(figsize=(10, 6))
plt.scatter(mean_distances, [1 if v == "RAG wins" else 0 for v in verdict_labels],
            c=colors, alpha=0.7, s=100, edgecolors='k')

plt.yticks([0, 1], ["No RAG better", "RAG better"])
plt.xlabel("Mean FAISS distance")
plt.ylabel("LLM Verdict")
plt.title("📊 Влияние расстояния FAISS на победу RAG vs No RAG")
plt.grid(True)
plt.tight_layout()
plt.show()