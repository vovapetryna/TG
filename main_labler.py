from vectorizing.vectorization import Vectorizer
from vectorizing.text_process import parse_article
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
model_path = os.path.join(dir, 'clustering', '__data__',  'model')
data_dir = os.path.join(dir, '__data__')

data_src = '/home/vova/PycharmProjects/TGmain/2703'

target_lang = 'ru'

vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                            pipe_en=pipe_en, pipe_ru=pipe_ru)

files = random.sample(list_files(data_src), 1000)

test_data = []

i = 0

for file in files:
    print('file id : %d' % i)
    if i % 100 == 0:
        save_object(test_data, os.path.join(data_dir, 'labels_ru'))
    i += 1

    vector, lang = vectorizer.vectorize_article_mean(file, word_limit=100)
    article = parse_article(open(file, "r"))
    if lang == target_lang:
        print('\033[1m' + article['title'] + '\033[0m')
        print('\033[1m' + article['description']+ '\033[0m')
        print('\033[1m' + article["p"][:100]+ '\033[0m\n')
        # test_data.append([list(vector), int(cluster)])
        try:
            type = int(input('1 - общество, 2 - экономика, 3 - технологии, 4 - спорт, 5 - развлечения, 6 - наука, 7 - другое, 8 - не новость\n'))
            test_data.append({"file": file,
                              "cluster": type})
            print('\n---------------------------------------------------------\n')
        except Exception as e:
            print('error')
            print(str(e))

# save_object(test_data, '/home/vova/PycharmProjects/TG/__data__/labels_load')
save_object(test_data, os.path.join(data_dir, 'labels_ru'))



