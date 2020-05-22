# https://scikit-learn.org/stable/modules/clustering.html#birch

from sklearn.cluster import Birch
import clustering.common as common
import pickle

class Birch_algo_wrapper:
    def __init__(self):
        self.wrapped = Birch()

    def fit(self,data):
        return self.wrapped.fit(data)

    def predict(self,data):
        return self.wrapped.predict(data)