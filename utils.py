import json
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib


def get_best_params(classifier_name, data_set_name, with_pca):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    params_dir = os.path.join(base_dir, 'params')
    pca_suffix = 'with_pca' if with_pca else 'without_pca'
    file_name = '%s_%s_%s.json' % (classifier_name, data_set_name, pca_suffix)
    file_path = os.path.join(params_dir, file_name)
    if not os.path.exists(file_path):
        err_msg = f'Path "{file_path}" does not exist. Run ./hypertune.py first.'
        raise ValueError(err_msg)
    with open(file_path, 'r') as file:
        return json.load(file)['best_params']


def load_classifier_with_best_params(classifier_name, data_set_name, with_pca):
    module = importlib.import_module(f'algorithms.{classifier_name}')
    classifier = module.get_classifier()
    best_params = get_best_params(classifier_name, data_set_name, with_pca)
    classifier.set_params(**best_params)
    return classifier


def plot_images(images, image_shape=(30, 40), image_transposed=True, n_rows=5, n_columns=10, fig_width=8.0):
    heights = image_shape[1]* np.ones(n_rows)

    sum_widths = (image_shape[0] * n_columns)
    fig_height = fig_width * sum(heights) / sum_widths # inches
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, sharex=True, sharey=True,
                             figsize=(fig_width, fig_height),
                             gridspec_kw={'height_ratios': heights})
    for row in range(n_rows):
        for column in range(n_columns):
            img = images[column + (row*n_columns)]
            img = img.reshape(image_shape)
            if image_transposed:
                img = img.T
            ax = axes[row][column]
            ax.imshow(img, cmap='gist_gray')
            ax.set_axis_off()
            ax.axis('off')
            ax.tick_params(axis='both', which='major', bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.subplots_adjust(wspace=0.0, hspace=0, left=0, right=1, bottom=0, top=1)
    return fig, axes
