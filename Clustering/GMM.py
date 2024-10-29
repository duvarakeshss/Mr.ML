# gmm.py

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class GMMClusteringModel:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=self.n_clusters)
        self.fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_metrics(self, X, labels):
        X_scaled = self.scaler.transform(X)
        metrics = {
            'Silhouette Score': silhouette_score(X_scaled, labels)
        }
        return metrics