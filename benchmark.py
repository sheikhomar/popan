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


def import_data_sets(data_set):
    if data_set == 'all':
        return [importlib.import_module(ds) for ds in DATA_SETS]
    if data_set not in DATA_SETS:
        msg = 'Unknown data set "%s"!' % data_set
        raise ap.ArgumentTypeError(msg)
    return [importlib.import_module(data_set)]


def import_algorithms(algorithm):
    if algorithm == 'all':
        return [importlib.import_module(f'algorithms.{al}') for al in ALGORITHMS]
    if algorithm in ALGORITHMS:
        return [importlib.import_module(f'algorithms.{algorithm}')]

    algorithms = []
    names = algorithm.split(',')
    for name in algorithm.split(','):
        if name not in ALGORITHMS:
            msg = 'Unknown algorithm "%s"!' % name
            raise ap.ArgumentTypeError(msg)
        module = importlib.import_module(f'algorithms.{name}')
        algorithms.append(module)
    return algorithms


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


def run_benchmark(algo, data_set, pca_components, n_folds, random_state, n_repeats, n_jobs):
    start_time = time.time()

    classifier_name = algo.__name__.replace('algorithms.', '')
    data_set_name = data_set.__name__

    print(f'Benchmarking {classifier_name} on {data_set_name} (PCA={pca_components})')

    print('Loading %s...' % data_set_name)
    X_train, X_test, y_train, y_test = data_set.load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    use_pca = pca_components is not None

    if pca_components is not None:
        print('Applying PCA...')
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X, y)

    best_params = fetch_best_params(classifier_name, data_set_name, use_pca)
    print(best_params)

    kfold = RepeatedStratifiedKFold(
        n_splits=n_folds,
        random_state=random_state,
        n_repeats=n_repeats
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
    benchmark_results = {
        'algorithm': classifier_name,
        'data_set': data_set_name,
        'pca': use_pca,
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
    pca_suffix = f'pca{pca_components}' if use_pca else 'original'
    file_name = '%s_%s_%s.json' % (data_set_name, pca_suffix, classifier_name)
    file_path = os.path.join('benchmark_results', file_name)
    write_json(benchmark_results, file_path)


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
    parser.add_argument('--threads', '-t',
                        required=False,
                        type=int,
                        default=-1,
                        help='Number of threads to utilise. For all available threads use -1.')
    parser.add_argument('--data-sets', '-d',
                        required=False,
                        type=import_data_sets,
                        default='all',
                        help=f'Supported values: all, {", ".join(DATA_SETS)}.')
    parser.add_argument('--algorithms', '-a',
                        required=False,
                        type=import_algorithms,
                        default='all',
                        help=f'Supported values: all, {", ".join(ALGORITHMS)}.')
    parser.add_argument('--pca', '-p',
                        required=False,
                        type=str,
                        default='all',
                        help=f'Supported values: all, original, 2d.')
    return parser.parse_args()


def main():
    args = parse_args()

    n_folds = args.folds
    random_state = args.random_state
    n_repeats = args.repeats
    n_jobs = args.threads

    print('Benchmark starting...')

    for ds in args.data_sets:
        for algorithm in args.algorithms:
            if args.pca in ['all', '2d']:
                run_benchmark(algo=algorithm, data_set=ds, pca_components=2, n_folds=n_folds,
                              random_state=random_state, n_repeats=n_repeats, n_jobs=n_jobs)
            if args.pca in ['all', 'original']:
                run_benchmark(algo=algorithm, data_set=ds, pca_components=None, n_folds=n_folds,
                              random_state=random_state, n_repeats=n_repeats, n_jobs=n_jobs)
    print('Benchmark done!')


if __name__ == '__main__':
    main()
