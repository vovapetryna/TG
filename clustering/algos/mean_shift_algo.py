# https://scikit-learn.org/stable/modules/clustering.html#mean-shift

from sklearn.cluster import MeanShift, estimate_bandwidth
import clustering.common as common
import pickle

class mean_shift_algo_wrapper:
    def __init__(self):
        self.wrapped = []
        self.data = []
        self.indexes =[]

    def fit(self, data):
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        self.wrapped = MeanShift()
        self.wrapped.fit(data)
        self.data = data
        self.indexes = self.wrapped.labels_

    def predict(self, data):
        return self.wrapped.predict(data)

model = mean_shift_algo_wrapper()

def do(input_data, draw_plot = False) -> common.AlgoInfo:
    global model
    model.fit(input_data)
    if draw_plot:
        common.draw(model.data, model.indexes)
    return common.AlgoInfo("Mean Shift", model.indexes)

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
