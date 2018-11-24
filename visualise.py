import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mnist as mnist

def pca_variance(X, max_d=100, highlights=[0.5, 0.7, 0.9], figsize=(13,8), figure_file_name=None):
    plt.rcParams.update({'font.size': 16})
    # 
    pca = PCA(n_components=max_d, random_state=42)
    pca.fit(X)

    # Plot the cumsum
    fig, ax = plt.subplots(figsize=(13, 8))
    x_coords = np.arange(1, max_d+1)
    y_coords = pca.explained_variance_ratio_.cumsum()
    ax.scatter(x_coords, y_coords, zorder=100, label='Cumulative sum')
    
    # Plot the normalised eigenvalue for each eigenvector
    ax.bar(x_coords, pca.explained_variance_ratio_, label='Normalised eigenvalue')

    # Highlight
    highlight_indicies = [1]
    for percent in highlights:
        highlight_index = np.where(y_coords > percent)[0][0]
        highlight_indicies.append(highlight_index)
    x_highlight = x_coords[highlight_indicies]
    y_highlight = y_coords[highlight_indicies]
    ax.vlines(x_highlight, 0, y_highlight, linestyles='dashed', color='#999999', zorder=1)
    ax.hlines(y_highlight, 0, x_highlight, linestyle='dashed', color='#999999', zorder=2)

    # Annotate highlights
    for xy in zip(x_highlight, y_highlight):
        ax.annotate('(%s, %.2f)' % xy, xy=xy,
                    textcoords='offset points', xytext=(20, -20),
                    zorder=200,
                    arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3")
                   )

    ax.legend(loc='center')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel('Number of eigenvectors')
    ax.set_ylabel('Normalised eigenvalue')
    ax.set_xlim(0, max_d+1)
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if figure_file_name is not None:
        fig.savefig(figure_file_name, bbox_inches='tight');


def mnist_2d(figure_file_name=None):
    plt.rcParams.update({'font.size': 14})
    
    # Load dataset
    X_train, X_test, y_train, y_test = mnist.load_data()
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    
    # Perform PCA transformation
    X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
              '#42d4f4', '#f032e6', '#bfef45', '#469990', '#e6beff',
              '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#a9a9a9']

    # Plot samples in each class separately
    for l in np.unique(y):
        ix = np.where(y == l)
        ax.scatter(X_2d[ix][:,0], X_2d[ix][:,1], 
                   label=l, marker='o', color=colors[l], s=40)
    ax.legend(ncol=2)
    ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    
    # Store the figure
    if figure_file_name is not None:
        fig.savefig(figure_file_name, bbox_inches='tight')


# 
def decision_boundary(classifier, X, y, step_size=0.02, figsize=(10, 8), figure_file_name=None):
    # Start by fitting the classifier
    classifier.fit(X, y)

    # https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
    # Generate mesh data
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step_size),
                           np.arange(x2_min, x2_max, step_size))
    mesh = np.c_[xx1.ravel(), xx2.ravel()]
    
    # Generate predictions based on the mesh data
    Z = classifier.predict(mesh)
    
    # Reshape results to fit the input shape that contour plot accepts.
    Z = Z.reshape(xx1.shape)

    # Predicted labels
    predicted_labels = list(np.unique(Z))
    
    leged_n_col = int(np.ceil(len(predicted_labels) / 20))
    
    # Plot the decision boundary
    fig, ax = plt.subplots(figsize=figsize)
    first_item = predicted_labels[0]-1
    plot = ax.contourf(xx1, xx2, Z, [first_item]+predicted_labels)
    
    # Plot legends
    legends, _ = plot.legend_elements()
    
    ax.legend(legends, predicted_labels, ncol=leged_n_col, loc="upper right", bbox_to_anchor=(1,1))
    ax.axis('off')
    if figure_file_name is not None:
        fig.savefig(figure_file_name, bbox_inches='tight')

