# import clustering.algo_manager as algo
# import clustering.common as common
# from json import loads
from vectorizing.vectorization import Vectorizer
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys
from clustering.algos import  DBSCAN_algo
from clustering.algos import agglomerative_algo
from clustering.algos import k_mean_algo
from clustering.algos import af_algo
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
temp_corp = os.path.join(dir, '__data__', 'temp_corp')
labeled_corp = os.path.join(dir, '__data__', 'temp_corp_labels')
labeled_corp_2 = os.path.join(dir, '__data__', 'temp_corp_labels_2')
test_set_file = os.path.join(dir, '__data__', 'test_set.json')

model_path = os.path.join(dir, 'clustering', '__data__',  'model')

"""LOAD TRAINING DATA"""
train_data = load_data(temp_corp)
target_lang = 'ru'
ru_corp = []
en_corp = []
zero_vector = list(np.zeros(300))
for vector, lang in train_data:
    if vector != zero_vector:
        if lang == 'ru':
            ru_corp.append(vector)
        elif lang == 'en':
            en_corp.append(vector)

corp = ru_corp if target_lang == 'ru' else en_corp
print('corpus size %d' % len(corp))
# corp = random.sample(corp, 1000)
# corp = corp[:30000]
# corp = random.sample(corp, 1000)
# test_no_labels = random.sample(corp, 10000)
test_no_labels = corp
"""CREATE TEST DATA"""
vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

test_data = load_data(test_set_file)
test_vectors = []
test_labels = []

for article in test_data:
    vector, lang = vectorizer.vectorize_article_mean(article["file"])
    if lang == target_lang:
        test_vectors.append(vector)
        test_labels.append(True if article["news_flag"] == 'news' else False)


"""fit models to the __data__"""
# algo.fit_models(__data__=corp, draw_result=False, save_folder=os.path.join(dir, 'clustering', 'models'))

# on this __data__ we will check if algorithm puts items in correspond clusters
#
# report = algo.compare_set_with_ideal_element(test_vectors, test_labels, models_folder=os.path.join(dir, 'clustering', 'models'))
# #
# #
# common.sortByTime(report)
# common.sortByRate(report)
#
# common.printAlgosInfo(report)

"""DBSCAN manual analyze"""
# algo = DBSCAN_algo.DBSCAN_algo_wrapper()
# report = algo.fit(corp)
#
# labels = algo.predict(test_no_labels)


"""Aglomerative algo"""
# algo = agglomerative_algo.AgglomerativeClustering_algo_wrapper()
# report = algo.fit(corp)
#
# labels = list(np.array(algo.predict(test_no_labels)).astype(int))


"""K_mean algo"""
algo = k_mean_algo.K_Means_wrapper()
report = algo.fit(corp)

labels = algo.predict(test_no_labels)

for vector in algo.wrapped.cluster_centers_:
    print(vectorizer.n_nearest(vector, target_lang, 10))

algo.save(model_path)

"""Affinity claster algo"""
# algo = af_algo.AffinityPropagation_algo_wrapper()
# report = algo.fit(cosine_similarity(corp))
#
# labels = algo.predict(cosine_similarity(test_no_labels))

#
# for vector in algo.wrapped.cluster_centers_:
#     print(vectorizer.n_nearest(vector, target_lang, 10))

# mean_clusters = {}
# for label, vector in zip(labels, test_no_labels):
#     vector = np.array(vector)
#     if mean_clusters.get(label) is None:
#         mean_clusters[label] = [vector, 1]
#     else:
#         mean_clusters[label][0] += vector
#         mean_clusters[label][1] += 1
#
# for key, value in mean_clusters.items():
#     mean_clusters[key][0] /= mean_clusters[key][1]
#     print(vectorizer.n_nearest(mean_clusters[key][0], target_lang))


"""gloabl save"""
lebeled_vectors = []
for vector, cluster in zip(test_no_labels, labels):
    lebeled_vectors.append((vector, int(cluster)))

save_object(lebeled_vectors, labeled_corp)



