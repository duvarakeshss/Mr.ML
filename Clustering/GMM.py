# gmm.py

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class GaussianClustering:
    def __init__(self, n_components=3):
        """
        Initialize the Gaussian Mixture Model (GMM) for clustering.
        
        :param n_components: Number of clusters (components) to fit.
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=self.n_components)
        self.fitted = False

    def fit(self, X):
        """
        Fit the GMM model to the dataset X.
        
        :param X: 2D numpy array of shape (n_samples, n_features)
        """
        # Standardize the data
        X_scaled = self.scaler.fit_transform(X)
        self.gmm.fit(X_scaled)
        self.fitted = True

    def predict(self, X):
        """
        Predict cluster labels for the dataset X.
        
        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: Cluster labels predicted by GMM
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict(X_scaled)

    def get_cluster_centers(self):
        """
        Get the cluster centers (means) of the GMM components.
        
        :return: Cluster centers
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        return self.scaler.inverse_transform(self.gmm.means_)

    def get_probabilities(self, X):
        """
        Get the soft clustering probabilities for each point in X.
        
        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: Probabilities of each point belonging to each cluster.
        """
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict_proba(X_scaled)
