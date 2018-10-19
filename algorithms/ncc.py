from sklearn.neighbors.nearest_centroid import NearestCentroid


def get_classifier():
    return NearestCentroid()


def get_params_space(data_set):
    return {
        'metric': ['cosine', 'euclidean', 'manhattan', 'braycurtis',
                   'canberra', 'chebyshev', 'seuclidean', 'sqeuclidean']
    }
