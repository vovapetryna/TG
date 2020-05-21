from vectorizing.vectorization import Vectorizer
from vectorizing.text_process import parse_article
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys
import time
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from clustering.algos import k_mean_algo

clusters = {1: [4, 10, 12, 0, 1, 3, 11, 7], 7: [19, 3, 4, 7, 5, 17, 13], 2: [2], 5: [6], 4: [8], 3: [16]}

def predict(cluster):
    ans = []
    for key, value in clusters.items():
        if cluster in value:
            ans.append(key)
    if len(ans) == 0:
        ans.append(1)
    return ans


dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
model_path = os.path.join(dir, 'clustering', '__data__',  'model')

data_dir = os.path.join(dir, '__data__')

data_src = '/home/vova/PycharmProjects/TGmain/2703'

target_lang = 'ru'

algo = k_mean_algo.K_Means_wrapper(src_data=model_path)
cluster_counts = len(algo.wrapped.cluster_centers_)
vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

gnb = MultinomialNB()

# files = random.sample(list_files(data_src), 2000)
labels_files = load_data(os.path.join(data_dir, 'labels_ru'))

clusters_data = {}

# test_data = []

start = time.time()
correct = 0

for example in labels_files:
    file = example["file"]
    vector, lang = vectorizer.vectorize_article_mean(file, word_limit=100)
    article = parse_article(open(file, "r"))
    if lang == target_lang:
        cluster = int(algo.predict([vector])[0])
        if clusters_data.get(example["cluster"]) is None:
            clusters_data[example["cluster"]] = [cluster]
        else:
            clusters_data[example["cluster"]].append(cluster)

        if example["cluster"] in predict(cluster):
            correct += 1
        else:
            print(example["cluster"], predict(cluster))
            print(article['title'])

print('for %d articles take %.2f sec' % (len(labels_files), time.time()-start))


# save_object(test_data, '/home/vova/PycharmProjects/TG/__data__/labels_load')
# save_object(clusters_data, 'log')

cluster_count = {}
res_clusters = {}

for key, value in clusters_data.items():
    for c in value:
        if cluster_count.get(c) is None:
            cluster_count[c] = 1
        else:
            cluster_count[c] += 1

for key, value in clusters_data.items():
    clusters_data[key] = Counter(value)
    for k, v in clusters_data[key].items():
        clusters_data[key][k] /= cluster_count[k]
        if clusters_data[key][k] > 0.33:
            if res_clusters.get(key) is None:
                res_clusters[key] = [k]
            else:
                res_clusters[key].append(k)

print(res_clusters)
print('total accuracy %.2f' % (correct/len(labels_files)))



