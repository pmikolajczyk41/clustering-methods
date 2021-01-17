import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def save_plots(plots, filename):
    pp = PdfPages(filename)
    for p in plots:
        pp.savefig(p)
    pp.close()

def plot_clusters(ax, x, ids, centroids=None):
    ax.scatter(*np.hsplit(x, 2), c=ids, cmap='gist_rainbow', alpha=.4)
    if centroids is not None:
        ax.plot(*np.hsplit(np.array(centroids), 2), 'o', ms=10, mfc='w', mew=2.5, mec='red')
