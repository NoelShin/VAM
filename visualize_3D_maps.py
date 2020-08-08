import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from time import time

from matplotlib.colors import to_rgb, to_rgba


def scatter(ax, x, y, z, color, alpha_arr, **kwarg):
    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    ax.scatter(x, y, z, c=color, **kwarg)
    return ax


def normalize(np_array):
    # np_array -= np.mean(np_array)
    # np_array = np_array / np.std(np_array)
    # np_array = 1 / (1 + np.exp(-np_array))
    np_array -= np_array.min()
    np_array = np_array / np_array.max()
    return np_array


def visualize_3d(np_array, dst):
    st = time()
    c, h, w = np_array.shape
    (X, Y, Z) = np.mgrid[0:c, 0:w, 0:h]
    top_percentage = 0.01

    # col = X.flatten() ** 2 + Y.flatten() ** 2 + Z.flatten()
    # col = np.random.rand(25 * 25 * 10)
    col = np_array.flatten()
    alpha_array = normalize(col)
    sorted_alpha = np.sort(alpha_array)  # Sort by ascending order
    print("length:", sorted_alpha.shape[0], "top n:", np.floor(sorted_alpha.shape[0] * top_percentage))
    print("criterion:", sorted_alpha[int(-np.floor(sorted_alpha.shape[0] * top_percentage))])
    top_n = sorted_alpha[int(-np.floor(sorted_alpha.shape[0] * top_percentage))]
    # top_n = sorted_alpha[(sorted_alpha.shape[0] - 1) - np.floor((sorted_alpha.shape[0] * top_percentage)).astype(np.uint32)]
    # print("top_n", top_n, "min", sorted_alpha[0])
    # exit(100)
    alpha_array = np.where(alpha_array < top_n, 0, alpha_array)
    print(alpha_array.min(), alpha_array.max())
    # col = softmax(col)
    # col = normalize(col)
    # alpha = col - col.min()
    # alpha = alpha / alpha.max()

    fig = plt.figure(1, figsize=(12, 8))
    fig.clf()
    ax = Axes3D(fig)
    cmap_color = "Greys"
    ax = scatter(ax, X, Y, Z, "k", alpha_array)
    # ax.scatter(X, Y, Z, c=col, cmap=cmap_color, alpha=0.1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")

    plt.draw()

    N = 21
    cmap = plt.get_cmap(cmap_color, N)
    norm = mpl.colors.Normalize(vmin=col.min(), vmax=col.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(col.min(), col.max(), N))
    plt.savefig(dst)
    print("Time taken:", time() - st)
    return
