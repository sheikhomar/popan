from sklearn.neighbors.classification import KNeighborsClassifier


def get_classifier():
    return KNeighborsClassifier()


def get_params_space(data_set):
    return {
        'n_neighbors': [1, 2, 3, 4, 5],
        'metric': ['euclidean', 'manhattan']
    }
