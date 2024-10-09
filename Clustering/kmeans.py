#k-mean clustering

import pandas as pd
import random
import numpy as np



class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.k)]
        for point in data:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            nearest_centroid = distances.index(min(distances))
            clusters[nearest_centroid].append(point)
        return clusters

    def update_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(random.choice(self.centroids))  # Handle empty cluster
        return new_centroids

    def has_converged(self, old_centroids, new_centroids):
        return np.array_equal(old_centroids, new_centroids)

    def fit(self, data):
        self.centroids = random.sample(list(data), self.k)

        for i in range(self.max_iterations):
            self.clusters = self.assign_clusters(data)
            new_centroids = self.update_centroids(self.clusters)

            if self.has_converged(self.centroids, new_centroids):
                print(f"Converged after {i + 1} iterations.")
                break

            self.centroids = new_centroids

        return self.centroids, self.clusters

    @staticmethod
    def read_csv(filename):
        data = pd.read_csv(filename)
        return data.values  # Convert DataFrame to a NumPy array
