# https://scikit-learn.org/stable/modules/clustering.html#mean-shift

from sklearn.cluster import MeanShift, estimate_bandwidth
import clustering.common as common
import pickle

class mean_shift_algo_wrapper:
    def __init__(self):
        self.wrapped = MeanShift()


    def fit(self, data):
        return self.wrapped.fit(data)

    def predict(self, data):
        return self.wrapped.predict(data)
