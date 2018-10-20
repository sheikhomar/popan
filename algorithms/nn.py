from sklearn.neighbors.classification import KNeighborsClassifier


def get_classifier():
    return KNeighborsClassifier()


def get_params_space(data_shape):
    n_samples, n_features = data_shape
    n_neighbours = [1, 2, 3, 4, 5]
    if n_features == 2 and n_samples > 10000:
        # Targets PCA-transformed MNIST data set
        n_neighbours = [1, 2, 3, 4, 5, 100, 300, 310, 320,
                        340, 340, 350, 360, 370, 380, 400]
    return {
        'n_neighbors': n_neighbours,
        'metric': ['euclidean', 'manhattan']
    }
