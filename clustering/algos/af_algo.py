# https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py

from sklearn.cluster import AffinityPropagation
import clustering.common as common
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class AffinityPropagation_algo_wrapper:
    def __init__(self, src_data=None):
        if src_data is None:
            self.wrapped = AffinityPropagation(damping=0.5, affinity="precomputed", convergence_iter=20)
        else:
            self.load(src_data)

    def fit(self,data):
        return self.wrapped.fit(cosine_similarity(data))

    def predict(self,data):
        return self.wrapped.fit_predict(cosine_similarity(data))

    def save(self, src_file):
        pickle.dump(self.wrapped, open(src_file, "wb"))

    def load(self, src_file):
        self.wrapped = pickle.load(open(src_file, "rb"))
