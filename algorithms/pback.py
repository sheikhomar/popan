# Perceptron trained using Backpropagation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np


class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.01, max_iterations=20):
        self.eta = eta
        self.max_iterations = max_iterations
        self.has_converged_ = True
        self.w_ = None

    def insert_ones(self, samples):
        ones = np.ones(samples.shape[1])
        return np.vstack([ones, samples])

    def calc_f(self, samples, labels, w):
        N = samples.shape[1]
        results = np.zeros(N)
        for i in range(N):
            x_i = samples[:, i].reshape(-1, 1)
            results[i] = labels[i] * np.asscalar(w.T.dot(x_i))
        return results

    def fit(self, X, y):
        self.w_ = None
        n_samples, n_features = X.shape
        w = np.random.rand(n_features+1)
        self.has_converged_ = False
        for it in range(self.max_iterations):
            response = self.calc_f(X, y, w)
            misclassified_filter = response < 0
            chi = X[:, misclassified_filter]
            num_misclassified = chi.shape[1]
            if num_misclassified == 0:
                self.has_converged_ = True
                break
            update_w = np.sum(y[misclassified_filter] * chi, axis=1).reshape(-1, 1)
            w = w + self.eta * update_w
        self.w_ = w
        return self

    def predict(self, X):
        check_is_fitted(self, 'w_')
        w = self.w_
        n_samples, n_features = X.shape
        results = np.zeros(n_samples)
        for i in range(n_samples):
            x_i = X[:, i].reshape(-1, 1)
            results[i] = np.asscalar(w.T.dot(x_i))
        labels = np.copysign(np.ones(n_samples), results)
        return labels
