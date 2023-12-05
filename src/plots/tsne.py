import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(data_tsne, labels, title, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels)
    plt.title(title)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    tsne = TSNE(n_components=4, random_state=42)

    test_participants = []
