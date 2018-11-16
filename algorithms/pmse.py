from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np


class MSEPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.weights_ = None
        self.classes_ = None

    def _augment(self, X):
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([ones, X], axis=1)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = self._augment(X)

        # Perform label encoding so label indicies start from zero
        le = LabelEncoder()
        encoded_y = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)

        # Construct a scaled version of the identity matrix
        eI = self.epsilon * np.eye(n_features+1)

        # Compute the pseudo-inverse
        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X) + eI), X.T)

        B = []
        for c_k in range(n_classes):
            b = np.where(encoded_y == c_k, 1, -1)
            B.append(b)

        # Store final result for prediction
        self.weights_ = np.dot(B, pseudo_inverse.T)

        return self

    def predict(self, X):
        check_is_fitted(self, 'weights_')

        # Retrieved trained weights
        W = self.weights_

        # Augment X
        X = self._augment(X)

        # Compute distance between the C decision function
        # and each of the samples in the test set
        distances = np.dot(X, W.T)

        # Classify by taking the class with the largest distance
        return self.classes_[np.argmax(distances, axis=1)]


def get_classifier():
    return MSEPerceptron()


def get_params_space(data_shape):
    return {
        'epsilon': [100, 10, 1, 0.1, 0.001, 0.0001]
    }
