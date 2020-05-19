# https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

from sklearn.cluster import AgglomerativeClustering
import clustering.common as common
import pickle

class AgglomerativeClustering_algo_wrapper:
    def __init__(self):
        self.wrapped = AgglomerativeClustering(linkage="ward", n_clusters=30, affinity='euclidean')
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self,data):
        return self.wrapped.fit_predict(data)

model = AgglomerativeClustering_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    global model
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("AgglomerativeClustering", model.indexes)

def predict(el) -> []:
    global model
    return model.predict(el)

def save(src):
    global model
    with open(src, "wb") as file:
        file.write(pickle.dumps(model, pickle.HIGHEST_PROTOCOL))
    del model

def load(src):
    global model
    pickle.load(open(src, "rb"))

def flush():
    global model
    del model
