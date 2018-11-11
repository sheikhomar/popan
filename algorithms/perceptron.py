from sklearn.linear_model import Perceptron


def get_classifier():
    return Perceptron(shuffle=True, max_iter=1000, tol=1e-3)


def get_params_space(data_shape):
    n_samples, n_features = data_shape
    return {
        'penalty': [None, 'l1', 'l2', 'elasticnet']
    }
