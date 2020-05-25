import sys
import os
from clustering.news_type import news_categories
from clustering.thread_former import news_thred
from vectorizing import lang_detector, vectorization
from json import dumps

# python tgnews languages <source_dir>
# python tgnews news <source_dir>
# python tgnews categories <source_dir>
# python tgnews threads <source_dir>

COMMAND_INDEX = 1
FOLDER_INDEX = 2

for i in range(len(sys.argv)):
    print("iteration : {}, arg: {}".format(i, sys.argv[i]))

def get_html_files(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.html' in file:
                result.append(os.path.join(root, file))

    for f in result:
        print(f)

    return result

def language_handle(file_names_list):
    detector = lang_detector.lang_detect()
    print(dumps(detector.detect_multiple_threads(file_names_list)))

def news_handle(file_names_list):
    dir = os.path.dirname(__file__)
    word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
    word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
    pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
    pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')

    model_ru = os.path.join(dir, 'clustering', '__data__', 'model_ru')
    model_en = os.path.join(dir, 'clustering', '__data__', 'model_en')

    vectorizer = vectorization.Vectorizer(pipe_en=pipe_en,
                                          model_file_en=word2vec_en,
                                          pipe_ru=pipe_ru,
                                          model_file_ru=word2vec_ru,
                                          restrict_vocab=200000, word_limit=100)

    n_c = news_categories(vectorizer, model_file_ru=model_ru,
                          model_file_en=model_en)

    print(dumps(n_c.predict_news(file_names_list)))

def categories_handle(file_names_list):
    dir = os.path.dirname(__file__)
    word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
    word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
    pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
    pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')

    model_ru = os.path.join(dir, 'clustering', '__data__', 'model_ru')
    model_en = os.path.join(dir, 'clustering', '__data__', 'model_en')

    vectorizer = vectorization.Vectorizer(pipe_en=pipe_en,
                                          model_file_en=word2vec_en,
                                          pipe_ru=pipe_ru,
                                          model_file_ru=word2vec_ru,
                                          restrict_vocab=200000, word_limit=100)

    n_c = news_categories(vectorizer, model_file_ru=model_ru,
                          model_file_en=model_en)

    print(dumps(n_c.predict_categories(file_names_list)))

def threads_handle(file_names_list):
    dir = os.path.dirname(__file__)
    word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
    word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
    pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
    pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')

    vectorizer = vectorization.Vectorizer(pipe_en=pipe_en,
                                          model_file_en=word2vec_en,
                                          pipe_ru=pipe_ru,
                                          model_file_ru=word2vec_ru,
                                          restrict_vocab=200000, word_limit=100)

    n_t = news_thred(vectorizer)

    print(dumps(n_t.form_thread(file_names_list)))

if len(sys.argv) < FOLDER_INDEX:
    exit(0)

input_filenames = get_html_files(sys.argv[FOLDER_INDEX])

if sys.argv[COMMAND_INDEX] == "language":
    language_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "news":
    news_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "categories":
    categories_handle(input_filenames)

if sys.argv[COMMAND_INDEX] == "threads":
    threads_handle(input_filenames)