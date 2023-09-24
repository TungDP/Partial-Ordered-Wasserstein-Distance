import matplotlib.pyplot as plt
import seaborn as sns
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


def soft_dtw_discrepancy(M):
    sdtw = SoftDTW(M, gamma=1)
    value = sdtw.compute()
    return value


def soft_dtw_viz(y1, y2, save_dir):
    D = SquaredEuclidean(y1.reshape(-1, 1), y2.reshape(-1, 1))
    sdtw = SoftDTW(D, gamma=1)
    sdtw.compute()

    E = sdtw.grad()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.invert_yaxis()
    sns.heatmap(E, ax=ax)

    plt.savefig(save_dir)
