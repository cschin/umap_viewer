#!/usr/bin/env python3
"""Generate a joined CSV for testing the Load CSV feature.

Output: test_joined.csv
Columns: id, x, y, labels, info, labels_shape
  - labels      : 5 clusters (cluster_A … cluster_E)
  - info        : ~1/3 markdown links, ~1/3 plain text, ~1/3 empty
  - labels_shape: secondary label set (geometric shape per cluster)
"""
import csv, math, random

random.seed(42)

clusters = [
    ("cluster_A", (2.0,  3.0), 0.6),
    ("cluster_B", (-3.0, 1.5), 0.5),
    ("cluster_C", (0.5, -2.5), 0.7),
    ("cluster_D", (4.0, -1.0), 0.4),
    ("cluster_E", (-2.0,-3.0), 0.6),
]

links = {
    "cluster_A": "https://en.wikipedia.org/wiki/Cluster_analysis",
    "cluster_B": "https://en.wikipedia.org/wiki/Dimensionality_reduction",
    "cluster_C": "https://en.wikipedia.org/wiki/UMAP",
    "cluster_D": "https://en.wikipedia.org/wiki/Machine_learning",
    "cluster_E": "https://en.wikipedia.org/wiki/Data_visualization",
}

# Secondary label set: assign a shape to each cluster
shapes = {
    "cluster_A": "circle",
    "cluster_B": "square",
    "cluster_C": "triangle",
    "cluster_D": "diamond",
    "cluster_E": "hexagon",
}

def gauss2(cx, cy, sigma):
    u1, u2 = random.random(), random.random()
    mag = sigma * math.sqrt(-2 * math.log(u1 + 1e-12))
    return cx + mag * math.cos(2 * math.pi * u2), cy + mag * math.sin(2 * math.pi * u2)

rows = []
n_per_cluster = 20
for label, (cx, cy), sigma in clusters:
    for i in range(n_per_cluster):
        pid = f"{label}_p{i:02d}"
        x, y = gauss2(cx, cy, sigma)
        rows.append((pid, x, y, label))

random.shuffle(rows)

out = "test_joined.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "x", "y", "labels", "info", "labels_shape"])
    for i, (pid, x, y, label) in enumerate(rows):
        r = i % 3
        if r == 0:
            info = f"[{label}]({links[label]})"
        elif r == 1:
            info = f"Point in {label} near ({x:.2f}, {y:.2f})"
        else:
            info = ""
        w.writerow([pid, f"{x:.6f}", f"{y:.6f}", label, info, shapes[label]])

print(f"Wrote {out}")
print(f"  {len(rows)} points, {len(clusters)} clusters")
print("  columns: id, x, y, labels, info, labels_shape")
print("  info: ~1/3 markdown links, ~1/3 plain text, ~1/3 empty")
