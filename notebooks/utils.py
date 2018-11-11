import numpy as np
import matplotlib.pyplot as plt

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
