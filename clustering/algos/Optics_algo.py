# https://scikit-learn.org/stable/modules/clustering.html#optics

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import clustering.common as common
import pickle

class OPTICS_algo_wrapper:
    def __init__(self):
        self.wrapped = OPTICS(min_samples=1, max_eps=2, metric='cosine', cluster_method='dbscan')

    def fit(self,data):
        return self.wrapped.fit(data)

    def predict(self,data):
        return self.wrapped.fit_predict(data)