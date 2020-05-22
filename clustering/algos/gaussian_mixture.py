# https://scikit-learn.org/stable/modules/mixture.html#mixture

from sklearn import mixture
import clustering.common as common
import pickle

class Gaussian_Mixture_algo_wrapper:
    def __init__(self):
        self.wrapped = mixture.GaussianMixture()

    def fit(self,data):
        return self.wrapped.fit(data)

    def predict(self,data):
        return self.wrapped.fit_predict(data)
