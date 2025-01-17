import importlib
nsc2 = importlib.import_module('algorithms.nsc2')


def get_classifier():
    return nsc2.NearestSubClassCentroid(n_subclasses=5)


def get_params_space(data_shape):
    return {
        'metric': ['euclidean', 'manhattan']
    }
