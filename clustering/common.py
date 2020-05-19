# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# general overview
# https://scikit-learn.org/stable/modules/clustering.html

import matplotlib.pyplot as plot
import numpy as np

colors = ('#FF0000', '#00FF00', '#0000FF', '#888888', '#4488FF', '#FF8844', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

class AlgoInfo:
    def __init__(self, algoname, clusters):
        self.clusters = clusters
        self.name = algoname
        self.time_taken = 0
        self.rate = 0
        self.toggle = False

    def __lt__(self, other):
        if self.toggle:
            return self.time_taken < other.time_taken
        else:
            return self.rate > other.rate

def sortByTime(data):
    for el in data:
        el.toggle = True
    data.sort()

def sortByRate(data):
    for el in data:
        el.toggle = False
    data.sort()

def printAlgosInfo(list):
    for item in list:
        print(" {:25s} | acuracy: {:3f} | [t: {:6f} sec] | ".format( item.name, item.rate, item.time_taken))

def draw(data, cluster_group):
    for i in range(len(data)):
        plot.scatter(data[i][0], data[i][1], 2, edgecolors=colors[int(cluster_group[i])])
    plot.show()

basic_test_n_1_two_claster_array_of_2d_points = [
    [5, 5],
    [4, 6],
    [6, 4],
    [4, 4],
    [6, 6],
    [3, 3],
    [2, 4],
    [4, 2],
    [2, 2],]
basic_test_n_1_diff = [True,True,True,True,True,False,False,False,False]
