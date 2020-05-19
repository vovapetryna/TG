# https://github.com/scikit-learn/scikit-learn/tree/483cd3eaa3c636a57ebb0dc4765531183b274df0/sklearn/cluster

from sklearn.cluster import KMeans
import clustering.common as common
import pickle

class K_Means_wrapper:
    def __init__(self):
        self.wrapped = KMeans(n_clusters=2)
        self.data = []
        self.indexes =[]

    def fit(self,data):
        self.wrapped.fit(data)
        self.indexes = self.wrapped.labels_
        self.data = data

    def predict(self,data):
        return self.wrapped.predict(data)

model = K_Means_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    global model
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("K_mean", model.indexes)


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
