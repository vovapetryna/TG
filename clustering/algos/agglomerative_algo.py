# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

from sklearn.cluster import AgglomerativeClustering
import clustering.common as common
import pickle

class AgglomerativeClustering_algo_wrapper:
    def __init__(self, scale):
        self.wrapped = AgglomerativeClustering(linkage="average", n_clusters=None, affinity='cosine',
                                               distance_threshold=scale)

    def fit(self, data):
        return self.wrapped.fit(data)

    def predict(self, data):
        return self.wrapped.fit_predict(data)

