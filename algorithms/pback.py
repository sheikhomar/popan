# Perceptron trained using Backpropagation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
import numpy as np


def _fit_binary_perceptron(X, y, pos_class, eta0=0.1, decay=0.01, max_iterations=1000):
    """
    Fits a single binary classifier.
    :param X: samples, matrix of the shape N * D
    :param y: labels, vector of size N
    :param pos_class: The positive class
    :return: the weight matrix
    """
    # Set positive class to 1 and the rest to -1
    y = np.where(y == pos_class, 1, -1)

    # Initial weight vector of size D
    w = np.random.rand(X.shape[1])

    has_converged = False
    n_misclassified = 0
    for iteration in range(max_iterations):
        # Compute the response of the decision function g()
        response = np.multiply(y, np.dot(X, w))

        # Construct chi; a matrix of misclassified samples
        misclassified_filter = response < 0
        chi = X[misclassified_filter, :]

        # Stop algorithm when all samples classified correctly
        n_misclassified = chi.shape[0]
        if n_misclassified == 0:
            has_converged = True
            break

        misclassified_y = y[misclassified_filter].reshape(-1, 1)
        update_w = np.sum(np.multiply(misclassified_y, chi), axis=0)
        learning_rate = eta0 * np.exp(-decay * iteration)
        # Perhaps faster alternative?
        # learning_rate = eta0 * (1. / (1. + decay * iteration))
        w = w + learning_rate * update_w

    if not has_converged:
        print('Waring: Maximum number of iteration reached before convergence. '
              'Consider increasing max_iterations to improve the fit. '
              'Number of misclassified samples: ' + str(n_misclassified))
    return w


class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, eta0=0.1, decay=0.01, max_iterations=1000, n_jobs=-1, verbose=2):
        self.eta0 = eta0
        self.decay = decay
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.weights_ = None
        self.classes_ = None

    def _augment(self, X):
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([ones, X], axis=1)

    def fit(self, X, y):
        # n_samples, n_features = X.shape
        X = self._augment(X)

        # Perform label encoding so label indicies start from zero
        le = LabelEncoder()
        encoded_y = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)

        # Use the Parallel library to fit C binary classifiers in parallel
        results = Parallel(
            n_jobs=self.n_jobs, prefer='threads', verbose=self.verbose
        )(delayed(_fit_binary_perceptron)(X, encoded_y, c, self.eta0, self.decay, self.max_iterations)
          for c in range(n_classes))

        # Store final result for prediction
        self.weights_ = np.array(results)

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
    return Perceptron()


def get_params_space(data_shape):
    return {
        'eta0': [0.001, 0.01, 0.1, 1],
        # When decay=0 then learning rate is fixed to eta0
        'decay': [0, 0.001, 0.01, 0.1, 1]
    }
