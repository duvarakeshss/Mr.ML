# spectral.py

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestCentroid

class SpectralClusteringModel:
    def __init__(self, n_clusters=3):
        """
        Initialize the Spectral Clustering model.
        
        :param n_clusters: Number of clusters to form.
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
        self.centroid_model = None
        self.fitted = False

    def fit(self, X):
        """
        Fit the Spectral Clustering model to the dataset X.
        
        :param X: 2D numpy array of shape (n_samples, n_features)
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        self.labels_ = self.model.fit_predict(X_scaled)
        
        # Calculate cluster centroids for prediction
        self.centroid_model = NearestCentroid()
        self.centroid_model.fit(X_scaled, self.labels_)
        
        self.fitted = True

    def predict(self, X):
        """
        Predict cluster labels for new data points.
        
        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: Predicted cluster labels for the new points.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        X_scaled = self.scaler.transform(X)
        return self.centroid_model.predict(X_scaled)

    def get_metrics(self, X, labels):
        """
        Calculate clustering evaluation metrics.
        
        :param X: The input features array (2D numpy array)
        :param labels: The predicted labels (1D numpy array)
        :return: Dictionary of clustering evaluation metrics.
        """
        X_scaled = self.scaler.transform(X)
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, labels),
            'Davies-Bouldin Index': davies_bouldin_score(X_scaled, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(X_scaled, labels)
        }
        return metrics
