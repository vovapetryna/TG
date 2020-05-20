# https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py

from sklearn.cluster import AffinityPropagation
import clustering.common as common
import pickle

class AffinityPropagation_algo_wrapper:
    def __init__(self):
        self.wrapped = AffinityPropagation(damping=0.9, affinity="precomputed")
        self.data = []
        self.indexes = []

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)
