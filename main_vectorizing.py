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
temp_corp_4 = os.path.join(dir, '__data__', 'temp_corp_4')

test_set_file = os.path.join(dir, '__data__', 'test_set.json')

files = list_files('/home/vova/PycharmProjects/TGmain/2703')
files = random.sample(files, 800000)
# files = random.sample(files, 3000)
corpus = []

vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

i = 0

"""THREAD CONFIG
Для того что-бы не парится я просто целиком проекты запускаю
так получаю разные потоки (это чисто для быстрой обработки большого корпуса)
Обработка 200000 в синг коре занимает 6 часов"""
# thread_file_src = temp_corp
# thread_file_src = temp_corp_2
# thread_file_src = temp_corp_3
thread_file_src = temp_corp_4

# files = files[:int(len(files)/4)]
# files = files[(int(len(files)/4)):(int(len(files)/4*2))]
# files = files[(int(len(files)/4*2)):(int(len(files)/4*3))]
files = files[(int(len(files)/4*3)):]

start = time.time()
for file in files:
    if i % 1000 == 0:
        print('article procesed %d with time %.2f' % (i, time.time() - start))
        start = time.time()
    i += 1

    if i % 10000 == 0:
        print('saving last data')
        with open(thread_file_src, "w") as f:
            f.write(dumps(list(corpus)))

    vector, lang = vectorizer.vectorize_article_mean(file, word_limit=100)

    if lang:
        corpus.append([list(vector), lang])

with open(thread_file_src, "w") as file:
    file.write(dumps(list(corpus)))
