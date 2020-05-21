# import clustering.algo_manager as algo
# import clustering.common as common
# from json import loads
from vectorizing.vectorization import Vectorizer
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys
from clustering.algos import af_algo
from clustering.algos import k_mean_algo
import time
import numpy as np

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
temp_corp = os.path.join(dir, '__data__', 'temp_corp_merge_2')
labeled_corp = os.path.join(dir, '__data__', 'temp_corp_labels')
labeled_corp_2 = os.path.join(dir, '__data__', 'temp_corp_labels_2')
test_set_file = os.path.join(dir, '__data__', 'test_set.json')

model_path = os.path.join(dir, 'clustering', '__data__',  'model_k_10')

vectorizer = Vectorizer(model_file_ru=word2vec_ru, model_file_en=word2vec_en,
                       pipe_en=pipe_en, pipe_ru=pipe_ru)

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
# corp = random.sample(corp, 10000)
# corp = corp[:30000]
# corp = random.sample(corp, 1000)
# test_no_labels = random.sample(corp, 10000)
test_no_labels = corp


"""fit models to the __data__"""

"""K_mean algo"""
algo = k_mean_algo.K_Means_wrapper()
report = algo.fit(corp)

labels = algo.predict(test_no_labels)

cluster_id = 0
for vector in algo.wrapped.cluster_centers_:
    print(cluster_id)
    cluster_id += 1
    print(vectorizer.n_nearest(vector, target_lang, 30))

algo.save(model_path)

"""Affinity claster algo"""
# algo = af_algo.AffinityPropagation_algo_wrapper()
# report = algo.fit(corp)
#
# labels = algo.predict(test_no_labels)
#
# algo.save(model_path)

"""gloabl save"""
lebeled_vectors = []
for vector, cluster in zip(test_no_labels, labels):
    lebeled_vectors.append((vector, int(cluster)))

save_object(lebeled_vectors, labeled_corp)



