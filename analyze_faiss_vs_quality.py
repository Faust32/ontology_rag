import json
import matplotlib.pyplot as plt

log_file = "rag_log.jsonl"

distances = []
verdicts = []

with open(log_file, "r") as f:
    for line in f:
        data = json.loads(line)
        if "mean_distance" in data and "verdict" in data:
            distances.append(data["mean_distance"])
            verdicts.append(data["verdict"])

colors = ["green" if "B" in v else "red" for v in verdicts]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(distances)), distances, c=colors, alpha=0.7)
plt.title("Связь между расстоянием FAISS и качеством ответа")
plt.xlabel("Запросы")
plt.ylabel("Средняя FAISS-дистанция")
plt.grid(True)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Answer B is better (RAG win)', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Answer A is better (no RAG)', markerfacecolor='red', markersize=10),
])
plt.tight_layout()
plt.show()
