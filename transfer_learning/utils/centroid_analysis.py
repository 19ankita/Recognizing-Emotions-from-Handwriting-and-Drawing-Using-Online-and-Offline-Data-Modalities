import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def plot_centroids_and_distances(X_2d, y, class_names, save_path):
    """
    X_2d: (N, 2) t-SNE embeddings
    y:    (N,) class labels
    """

    centroids = {}

    # Compute centroids
    for idx, name in enumerate(class_names):
        centroids[name] = X_2d[y == idx].mean(axis=0)

    # Plot points
    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(class_names):
        plt.scatter(
            X_2d[y == idx, 0],
            X_2d[y == idx, 1],
            label=name,
            alpha=0.4,
            s=20
        )

    # Plot centroids
    for name, c in centroids.items():
        plt.scatter(c[0], c[1], marker="X", s=200, edgecolor="k")
        plt.text(c[0], c[1], name, fontsize=10, weight="bold")

    plt.legend()
    plt.title("t-SNE with Class Centroids")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Print centroid distances
    print("\nCentroid distances:")
    centroid_matrix = np.vstack(list(centroids.values()))
    dist = pairwise_distances(centroid_matrix)

    for i, ci in enumerate(class_names):
        for j, cj in enumerate(class_names):
            if i < j:
                print(f"{ci} ↔ {cj}: {dist[i, j]:.2f}")
