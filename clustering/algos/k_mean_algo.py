# https://github.com/scikit-learn/scikit-learn/tree/483cd3eaa3c636a57ebb0dc4765531183b274df0/sklearn/cluster

from sklearn.cluster import KMeans
import clustering.common as common
import pickle

class K_Means_wrapper:
    def __init__(self, src_data=None):
        if src_data is None:
            self.wrapped = KMeans(n_clusters=10)
        else:
            self.load(src_data)

    def fit(self,data):
        self.wrapped.fit(data)

    def predict(self,data):
        return self.wrapped.predict(data)

    def save(self, src_file):
        pickle.dump(self.wrapped, open(src_file, "wb"))

    def load(self, src_file):
        self.wrapped = pickle.load(open(src_file, "rb"))