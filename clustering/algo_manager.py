# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from clustering.algos import\
    DBSCAN_algo as dbscan, \
    k_mean_algo as k_mean, \
    af_algo as af, \
    agglomerative_algo as ag, \
    Optics_algo as optics, \
    Birch as birch, \
    gaussian_mixture as gaussian, \
    mean_shift_algo as mean_shift, \
    spectral as spectral

import random
import numpy as np
import clustering.common as common
from timeit import default_timer as timer
import os

algo_model = [af.model, birch.model, mean_shift.model, optics.model, ag.model, dbscan.model, k_mean.model]
algo_list = [af.do, birch.do, mean_shift.do, optics.do, ag.do, dbscan.do, k_mean.do]
algo_predict_list = [af.predict, birch.predict, mean_shift.predict, optics.predict, ag.predict, dbscan.predict, k_mean.predict]
algo_save = [af.save, birch.save, mean_shift.save, optics.save, ag.save, dbscan.save, k_mean.save]
algo_load = [af.load, birch.load, mean_shift.load, optics.load, ag.load, dbscan.load, k_mean.load]
algo_flush = [af.flush, birch.flush, mean_shift.flush, optics.flush, ag.flush, dbscan.flush, k_mean.flush]
algo_name = ['af', 'birch', 'mean_shift', 'optics', 'ag', 'dbscan', 'k_mean']

def fit_models(data, draw_result, save_folder):
    for i in range(len(algo_list)):
        print(algo_name[i])
        algo_list[i](data, draw_result)
        algo_save[i](os.path.join(save_folder, algo_name[i]))

def analyze(data, draw_results) -> []:
    report = []
    for i in range(len(algo_list)):
        print(algo_name[i])
        start = timer()
        info = algo_list[i](data, draw_results)
        end = timer()
        info.time_taken = end - start
        report.append(info)
    return report

def generate_test_data(clusters_num, points_in_cluster) -> []:
    data = [[]]
    for cluster in range(clusters_num):
        x0 = random.random()
        y0 = random.random()
        for i in range(points_in_cluster):
            x = random.random() ** 2
            y = random.random() ** 2
            data.append([(x - x0) ** 2, (y - y0) ** 2])
    return np.array(data[1:])

def compare_set_with_ideal_element(set, diff, models_folder) -> []:
    reports = []
    for i in range(len(algo_predict_list)):
        reports.append(common.AlgoInfo("", []))

    dividor = len(set)

    for i in range(len(algo_predict_list)):
        algo_load[i](os.path.join(models_folder, algo_name[i]))
        start = timer()
        try:
            clusters_result = algo_predict_list[i](set)
            reports[i].time_taken = timer() - start
            reports[i].clusters = algo_model[i].indexes
            reports[i].name = algo_name[i]
            algo_flush[i]()
            for j in range(len(set)):
                if diff[j] == (clusters_result[0] == clusters_result[j]):
                    reports[i].rate = reports[i].rate + 1
            reports[i].rate = reports[i].rate / dividor
        except Exception as e:
            print('problem with %s' % algo_name[i])
            print('problem is %s' % str(e))


    return reports
