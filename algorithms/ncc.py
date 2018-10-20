from sklearn.neighbors.nearest_centroid import NearestCentroid


def get_classifier():
    return NearestCentroid()


def get_params_space(data_shape):
    return {
        # TODO: Investigate how to compute the centroid for other distance metrics.
        # scikit-learn supports other distances like 'braycurtis', 'canberra',
        # 'chebyshev', 'seuclidean', 'sqeuclidean' but is it okay to use the mean
        # to compute the centroids? NearestCentroid issues warnings when the other
        # distance metrics are used. Can we safely ignore these warnings?
        'metric': ['euclidean', 'manhattan']
    }
