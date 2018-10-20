#!/usr/bin/env python
import argparse as ap
import os
from os import path
import importlib
import json
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


ALGORITHMS_DIR = 'algorithms'

ALGORITHMS = [path.splitext(f)[0]
              for f in os.listdir(ALGORITHMS_DIR)
              if path.isfile(path.join(ALGORITHMS_DIR, f))]
DATA_SETS = ['mnist', 'orl']


def import_algorithm(algorithm):
    if algorithm not in ALGORITHMS:
        msg = 'Unknown algorithm "%s"!' % algorithm
        raise ap.ArgumentTypeError(msg)
    return importlib.import_module('algorithms.%s' % algorithm)


def import_data_set(data_set):
    if data_set not in DATA_SETS:
        msg = 'Unknown data set "%s"!' % data_set
        raise ap.ArgumentTypeError(msg)
    return importlib.import_module(data_set)


def parse_args():
    parser = ap.ArgumentParser(
        description='Determines the best hyper-parameters for '
                    'the given algorithm.')
    parser.add_argument('--algorithm', '-a',
                        type=import_algorithm,
                        required=True,
                        help='Supported algorithms: %s' % ', '.join(ALGORITHMS))
    parser.add_argument('--data-set', '-d',
                        type=import_data_set,
                        required=True,
                        help='Supported data sets: %s' % ', '.join(DATA_SETS))
    parser.add_argument('--pca',
                        action='store_true',
                        required=False,
                        default=False,
                        help='Reduce dimensions using PCA')
    parser.add_argument('--folds', '-k',
                        required=False,
                        default=5,
                        help='Number of folds or splits used in the k-fold cross validation.')
    return parser.parse_args()


def save_results(searcher, algo_name, data_set_name, with_pca):
    num_folds = searcher.cv
    search_results = searcher.cv_results_
    num_params = len(search_results['params'])
    test_scores = np.zeros((num_folds, num_params))
    for i in range(num_folds):
        for j in range(num_params):
            test_scores[i][j] = search_results['split%d_test_score' % i][j]
    test_scores = test_scores.transpose()

    output = {
        'algorithm': algo_name,
        'data_set': data_set_name,
        'pca': with_pca,
        'best_score': searcher.best_score_,
        'best_params': searcher.best_params_,
        'folds': searcher.cv,
        'search_params': search_results['params'],
        'test_scores': test_scores.tolist(),
    }

    pca_suffix = 'with_pca' if with_pca else 'without_pca'
    results_file_name = '%s_%s_%s.json' % (algo_name, data_set_name, pca_suffix)
    result_file_path = os.path.join('params', results_file_name)
    with open(result_file_path, 'w') as results_file:
        json.dump(output, results_file, sort_keys=False, indent=4, separators=(',', ': '))

    print('Output: ')
    print(output)


def main():
    args = parse_args()
    num_folds = args.folds
    algo = args.algorithm
    algo_name = algo.__name__.replace('algorithms.', '')
    data_set = args.data_set
    data_set_name = data_set.__name__
    with_pca = args.pca

    print('Loading %s...' % data_set_name)
    X_train, X_test, y_train, y_test = data_set.load_data()

    if with_pca:
        print('Applying PCA...')
        pca = PCA(n_components=2)
        pca.fit(X_train, y_train)
        X_train = pca.transform(X_train)

    print('Using algorithm %s...' % algo_name)
    classifier = algo.get_classifier()
    params_space = algo.get_params_space(X_train.shape)
    searcher = GridSearchCV(
        classifier,
        params_space,
        cv=num_folds,
        n_jobs=-1,
        scoring='accuracy',
        verbose=2,
        iid=False,
        return_train_score=False
    )


    print('Running grid search...')
    searcher.fit(X_train, y_train)
    print('Saving results...')
    save_results(searcher, algo_name, data_set_name, with_pca)
    print('Done!')


if __name__ == '__main__':
    main()
