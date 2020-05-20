import umap
from sys_tools.preprocess import load_data, save_object
import random
import numpy as np
import matplotlib.pyplot as plt

colors = ('#FF0000', '#00FF00', '#0000FF', '#888888', '#4488FF', '#FF8844', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

class umap_reduse:
    def __init__(self, n_neighbors, min_dist, metric='cosine'):
        self.model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                               n_components=2, random_state=42)

    def fit(self, vectors):
        return self.model.fit_transform(list(vectors))


def main():
    temp_corp = '/home/vova/PycharmProjects/TG/__data__/labels_load'
    train_data = np.array(load_data(temp_corp))
    vectors = train_data[:,0]
    labels = train_data[:,1]
    # for i in range(len(labels)):
    #     labels[i] = colors[labels[i]]
    print(vectors[:10])
    print(labels[:10])

    u_reduse = umap_reduse(30, 0.0)

    data = np.array(u_reduse.fit(vectors))
    plt.scatter(data[:,0], data[:,1], c=labels, s=3)
    plt.show()


if __name__ == "__main__":
    main()