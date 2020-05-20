from vectorizing.vectorization import Vectorizer
from vectorizing.text_process import parse_article
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys

from clustering.algos import k_mean_algo

prediction_weight ={"Общество": [6],
               "экономика": [5],
               "технологии": [],
               "спорт": [9],
               "развлечение": [2, 8],
               "наука": [],
               "другое": [7],
               "не новость": [4]}

def predict(cluster):
    prediction = ""
    for key, value in prediction_weight.items():
        if cluster in value:
            prediction += " " + key
    return prediction

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
model_path = os.path.join(dir, 'clustering', '__data__',  'model')

data_src = '/home/vova/PycharmProjects/TGmain/2703'

target_lang = 'ru'

algo = k_mean_algo.K_Means_wrapper(src_data=model_path)
vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

files = random.sample(list_files(data_src), 200)

clusters_data = {}

# test_data = []

for file in files:
    vector, lang = vectorizer.vectorize_article_mean(file, word_limit=100)
    article = parse_article(open(file, "r"))
    if lang == target_lang:
        cluster = algo.predict([vector])[0]
        print(article['title'])
        print(article['description'])
        print('ЭТО КАТЕГОРИЯ : %s ' % predict(cluster))
        # test_data.append([list(vector), int(cluster)])
        try:
            type = int(input('общество - 1, экономика -2, технологии - 3, спорт -4, развлечения -5, наука-6, другое-7, не новость -8'))
            if clusters_data.get(type) is None:
                clusters_data[int(type)] = [int(cluster)]
            else:
                clusters_data[int(type)].append(int(cluster))
        except Exception as e:
            print('error')
            print(str(e))

# save_object(test_data, '/home/vova/PycharmProjects/TG/__data__/labels_load')
save_object(clusters_data, 'log')



