# https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py

from sklearn.cluster import AffinityPropagation
import clustering.common as common
import pickle

class AffinityPropagation_algo_wrapper:
    def __init__(self):
        self.wrapped = AffinityPropagation()
        self.data = []
        self.indexes = []

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

model = AffinityPropagation_algo_wrapper()

def do(input_data, draw_plot=False) -> common.AlgoInfo:
    global model
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("AffinityPropagation", model.indexes)

def load(src):
    global model
    pickle.load(open(src, "rb"))

def flush():
    global model
    del model

def save(src):
    global model
    with open(src, "wb") as file:
        file.write(pickle.dumps(model, pickle.HIGHEST_PROTOCOL))
    del model

def predict(el) -> []:
    global model
    return model.predict(el)