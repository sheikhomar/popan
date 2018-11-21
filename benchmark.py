#!/usr/bin/env python
import argparse as ap
import os
from os import path
import importlib
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import time


ALGORITHMS_DIR = 'algorithms'

ALGORITHMS = [path.splitext(f)[0]
              for f in os.listdir(ALGORITHMS_DIR)
              if path.isfile(path.join(ALGORITHMS_DIR, f))]
DATA_SETS = ['orl', 'mnist']


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


def benchmark(algo, X, y, data_set_name, with_pca, n_folds, random_state, n_iterations, n_jobs=-1):
    start_time = time.time()

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
    print('X={}  y={}'.format(X.shape, y.shape))
    if classifier_name in ['pmse', 'pback']:
        n_jobs = 2  # TODO: Avoid memory issues.
    final_scores = cross_val_score(
        classifier, X, y, n_jobs=n_jobs, cv=kfold, scoring='accuracy', verbose=2
    )

    print(final_scores)

    exec_time_sec = (time.time() - start_time)
    return {
        'algorithm': classifier_name,
        'data_set': data_set_name,
        'pca': with_pca,
        'params': best_params,
        'execution_time_sec': exec_time_sec,
        'n_jobs': n_jobs,
        'scores_summary': {
            'min': final_scores.min(),
            'mean': final_scores.mean(),
            'max': final_scores.max(),
            'variance': final_scores.var(ddof=1),
            'std': final_scores.std(ddof=1),
        },
        'scores': list(final_scores)
    }


def run(args, data_set, with_pca):
    data_set_name = data_set.__name__

    print('Loading %s...' % data_set_name)
    X_train, X_test, y_train, y_test = data_set.load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if with_pca:
        print('Applying PCA...')
        pca = PCA(n_components=2)
        X = pca.fit_transform(X, y)

    n_folds = args.folds
    random_state = args.random_state
    n_repeats = args.repeats

    for module_name in ALGORITHMS:
        pca_suffix = 'pca' if with_pca else 'original'
        file_name = '%s_%s_%s.json' % (data_set_name, pca_suffix, module_name)
        file_path = os.path.join('benchmark_results', file_name)

        if os.path.exists(file_path):
            print('Benchmark data already exists for {}!'.format(module_name))
            continue

        algo = importlib.import_module('algorithms.%s' % module_name)
        results = benchmark(algo, X, y, data_set_name, with_pca,
                            n_folds, random_state, n_repeats)
        write_json(results, file_path)


def parse_args():
    parser = ap.ArgumentParser(
        description='Benchmarks the different classifiers.')
    parser.add_argument('--random-state', '-r',
                        required=False,
                        type=int,
                        default=42,
                        help='Random state for consistent results.')
    parser.add_argument('--folds', '-k',
                        required=False,
                        type=int,
                        default=3,
                        help='Number of folds or splits used in the k-fold cross validation.')
    parser.add_argument('--repeats', '-n',
                        required=False,
                        type=int,
                        default=33,
                        help='Number of times to repeat the cross-validation.')
    return parser.parse_args()


def main():
    print('Benchmarking')
    args = parse_args()
    data_sets = [importlib.import_module(ds) for ds in DATA_SETS]
    for ds in data_sets:
        run(args, data_set=ds, with_pca=True)
        run(args, data_set=ds, with_pca=False)
    print('Benchmark done!')


if __name__ == '__main__':
    main()
