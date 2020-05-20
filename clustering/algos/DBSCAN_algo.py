# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

from sklearn.cluster import DBSCAN
import clustering.common as common
import pickle

class DBSCAN_algo_wrapper:
    def __init__(self):
        self.wrapped = DBSCAN(eps=10, min_samples=100, algorithm='brute')
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

