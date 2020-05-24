# https://scikit-learn.org/stable/modules/clustering.html#birch

from sklearn.cluster import Birch
import clustering.common as common
import pickle

class Birch_algo_wrapper:
    def __init__(self):
        self.wrapped = Birch(n_clusters=None, threshold=0.5, branching_factor=50)

    def fit(self,data):
        return self.wrapped.fit(data)

    def fit_predict(self,data):
        self.wrapped = self.wrapped.partial_fit(data)
        return self.wrapped.predict(data)

    def predict(self, data):
        return self.wrapped.predict(data)