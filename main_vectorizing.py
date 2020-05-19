# import clustering.algo_manager as algo
# import clustering.common as common
from json import loads, dumps
from vectorizing.vectorization import Vectorizer
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys
from clustering.algos import DBSCAN_algo
import time

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
temp_corp = os.path.join(dir, '__data__', 'temp_corp')
temp_corp_2 = os.path.join(dir, '__data__', 'temp_corp_2')
temp_corp_3 = os.path.join(dir, '__data__', 'temp_corp_3')
test_set_file = os.path.join(dir, '__data__', 'test_set.json')

files = list_files('/home/vova/PycharmProjects/TGmain/2703')
files = random.sample(files, 100000)
corpus = []

vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

i = 0
start = time.time()
for file in files:
    if i % 1000 == 0:
        print('article procesed %d with time %.2f' % (i, time.time() - start))
        start = time.time()
    i += 1

    vector, lang = vectorizer.vectorize_article_mean(file, word_limit=200)

    if lang:
        corpus.append([list(vector), lang])

with open(temp_corp_3, "w") as file:
    file.write(dumps(list(corpus)))