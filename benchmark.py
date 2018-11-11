#!/usr/bin/env python
import argparse as ap
import os
from os import path
import importlib
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

ALGORITHMS_DIR = 'algorithms'

ALGORITHMS = [path.splitext(f)[0]
              for f in os.listdir(ALGORITHMS_DIR)
              if path.isfile(path.join(ALGORITHMS_DIR, f))]
DATA_SETS = ['mnist', 'orl']


def import_algorithm(algorithm):
    #print('Given : "{}"'.format(algorithm))
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
                        type=int,
                        default=5,
                        help='Number of folds or splits used in the k-fold cross validation.')
    parser.add_argument('--random-state', '-r',
                        required=False,
                        type=int,
                        default=42,
                        help='Random state for consistent results.')
    parser.add_argument('--repeats', '-n',
                        required=False,
                        type=int,
                        default=42,
                        help='Number of times to repeat the cross-validation.')
    return parser.parse_args()


def get_result_path(subdir, classifier_name, data_set_name, with_pca):
    pca_suffix = 'with_pca' if with_pca else 'without_pca'
    file_name = '%s_%s_%s.json' % (classifier_name, data_set_name, pca_suffix)
    return os.path.join(subdir, file_name)


def write_json(obj, file_path):
    with open(file_path, 'w') as file:
        json.dump(obj, file, sort_keys=False, indent=4, separators=(',', ': '))


def fetch_best_params(classifier_name, data_set_name, with_pca):
    file_path = get_result_path('params', classifier_name, data_set_name, with_pca)
    if not os.path.exists(file_path):
        err_msg = 'Path "%s" does not exist. Run ./hypertune.py first.' % file_path
        raise ValueError(err_msg)
    with open(file_path, 'r') as file:
        return json.load(file)['best_params']


def save_results(results, classifier_name, data_set_name, with_pca, hyperparams):
    output = {
        'algorithm': classifier_name,
        'data_set': data_set_name,
        'pca': with_pca,
        'params': hyperparams,
        'scores': list(results)
    }
    file_path = get_result_path('benchmark_results', classifier_name, data_set_name, with_pca)
    write_json(output, file_path)


def benchmark(algo, X, y, data_set_name, with_pca, n_folds, random_state, n_iterations):
    classifier_name = algo.__name__.replace('algorithms.', '')

    print('Benchmarking %s' % classifier_name)

    best_params = fetch_best_params(classifier_name, data_set_name, with_pca)
    print(best_params)

    kfold = RepeatedStratifiedKFold(
        n_splits=n_folds,
        random_state=random_state,
        n_repeats=n_iterations
    )
    classifier = algo.get_classifier()
    classifier.set_params(**best_params)
    final_scores = cross_val_score(
        classifier, X, y, n_jobs=-1, cv=kfold, scoring='accuracy', verbose=2
    )

    print(final_scores)
    return {
        'algorithm': classifier_name,
        'data_set': data_set_name,
        'pca': with_pca,
        'params': best_params,
        'scores': list(final_scores)
    }


def main():
    print('Benchmarking')
    args = parse_args()
    n_folds = args.folds
    data_set = args.data_set
    data_set_name = data_set.__name__
    with_pca = args.pca
    random_state = args.random_state

    print('Loading %s...' % data_set_name)
    X_train, X_test, y_train, y_test = data_set.load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if with_pca:
        print('Applying PCA...')
        pca = PCA(n_components=2)
        pca.fit(X, y)
        X = pca.transform(X_train)

    final_output = []
    ALGORITHMS.remove('pmse')
    ALGORITHMS.remove('pback')
    for module_name in ALGORITHMS:
        algo = importlib.import_module('algorithms.%s' % module_name)
        res = benchmark(algo, X, y, data_set_name, with_pca, n_folds, random_state, 10)
        final_output.append(res)

    pca_suffix = 'with_pca' if with_pca else 'without_pca'
    file_name = '%s_%s.json' % (data_set_name, pca_suffix)
    file_path = os.path.join('benchmark_results', file_name)
    write_json(final_output, file_path)


if __name__ == '__main__':
    main()
