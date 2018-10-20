import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances


class NearestSubClassCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='euclidean', n_subclasses=1):
        self.metric = metric
        self.n_subclasses = n_subclasses
        self.classes_ = []
        self.centroids_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Perform label encoding so label indicies start from zero
        le = LabelEncoder()
        encoded_y = le.fit_transform(y)

        self.classes_ = le.classes_
        n_classes = len(self.classes_)
        self.centroids_ = np.empty((n_classes, self.n_subclasses, n_features))

        for i in range(n_classes):
            class_mask = encoded_y == i
            kmeans = KMeans(n_clusters=self.n_subclasses)
            kmeans.fit(X[class_mask], y[class_mask])
            self.centroids_[i] = kmeans.cluster_centers_
        return self

    def predict(self, X):
        check_is_fitted(self, 'centroids_')
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        distances = np.zeros((n_classes, n_samples))
        for i in range(n_classes):
            # Compute the distance between each sample and each centroid (n_samples, n_subclasses)
            pair_dist = pairwise_distances(X, self.centroids_[i], metric=self.metric)
            # For each sample, find the distance of the centroid with the lowest distance
            distances[i] = np.min(pair_dist, axis=1)
        return self.classes_[np.argmin(distances, axis=0)]


def get_classifier():
    return NearestSubClassCentroid(n_subclasses=2)


def get_params_space(data_set):
    return {
        'metric': ['euclidean', 'manhattan']
    }
