#!/usr/bin/env python3
"""Generate mock UMAP test CSVs with coords, labels, and an info column."""
import csv, math, random

random.seed(42)

# Five clusters at fixed centers
clusters = [
    ("cluster_A", (2.0,  3.0), 0.6),
    ("cluster_B", (-3.0, 1.5), 0.5),
    ("cluster_C", (0.5, -2.5), 0.7),
    ("cluster_D", (4.0, -1.0), 0.4),
    ("cluster_E", (-2.0,-3.0), 0.6),
]

# Sample web links per cluster for the info column
links = {
    "cluster_A": "https://en.wikipedia.org/wiki/Cluster_analysis",
    "cluster_B": "https://en.wikipedia.org/wiki/Dimensionality_reduction",
    "cluster_C": "https://en.wikipedia.org/wiki/UMAP",
    "cluster_D": "https://en.wikipedia.org/wiki/Machine_learning",
    "cluster_E": "https://en.wikipedia.org/wiki/Data_visualization",
}

def gauss2(cx, cy, sigma):
    # Box-Muller
    u1, u2 = random.random(), random.random()
    mag = sigma * math.sqrt(-2 * math.log(u1 + 1e-12))
    return cx + mag * math.cos(2 * math.pi * u2), cy + mag * math.sin(2 * math.pi * u2)

points = []
n_per_cluster = 20
for label, (cx, cy), sigma in clusters:
    for i in range(n_per_cluster):
        pid = f"{label}_p{i:02d}"
        x, y = gauss2(cx, cy, sigma)
        points.append((pid, x, y, label))

random.shuffle(points)

# Write coords CSV
with open("test_coords.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "x", "y"])
    for pid, x, y, _ in points:
        w.writerow([pid, f"{x:.6f}", f"{y:.6f}"])

# Write labels CSV with mixed info:
#   every 3rd point  -> markdown link  [ClusterName](url)
#   every 3rd+1      -> plain text description
#   every 3rd+2      -> empty (no info)
with open("test_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "labels", "info"])
    for i, (pid, x, y, label) in enumerate(points):
        r = i % 3
        if r == 0:
            info = f"[{label}]({links[label]})"
        elif r == 1:
            info = f"Point in {label} near ({x:.2f}, {y:.2f})"
        else:
            info = ""
        w.writerow([pid, label, info])

print("Wrote test_coords.csv and test_labels.csv")
print(f"  {len(points)} points across {len(clusters)} clusters")
print("  info column: ~1/3 markdown links, ~1/3 plain text, ~1/3 empty")
