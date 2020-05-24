# import clustering.algo_manager as algo
# import clustering.common as common
from json import loads, dumps
from vectorizing.vectorization import Vectorizer
from sys_tools.preprocess import list_files, load_data, save_object
import random
import os, sys
from clustering.algos import DBSCAN_algo
import time
import pickle


def main():

    dir = os.path.dirname(__file__)
    word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
    word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
    pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
    pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
    temp_corp = os.path.join(dir, '__data__', 'temp_corp')

    test_set_file = os.path.join(dir, '__data__', 'test_set.json')

    files = list_files('/home/vova/PycharmProjects/TGmain/2703')
    files = random.sample(files, 200000)
    vectorizer = Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                                pipe_en=pipe_en, pipe_ru=pipe_ru, restrict_vocab=200000, word_limit=100)

    i = 0

    vecs, _ = vectorizer.vectorize_multiple_files_multi(files)

    with open(temp_corp, "wb") as f:
        f.write(pickle.dumps(vecs, pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()