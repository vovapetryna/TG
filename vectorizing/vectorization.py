import gensim
import time
import numpy as np
from vectorizing.text_process import TextProcess
from multiprocessing import Pool, Process, Manager
from math import ceil
from sys_tools.preprocess import list_files, read_files_to_queue
import queue
import pickle
import random


def slice_list(list, parts=4):
    part_len = ceil(len(list) / parts)
    
    return [list[i:i+part_len] for i in range(0, len(list), part_len)]

class Vectorizer:
    def __init__(self, model_file_ru, model_file_en, pipe_ru, pipe_en, restrict_vocab=500000, word_limit=100):
        self.word_limit = word_limit
        self.restrict_vocab = restrict_vocab
        print('vectorizer start loading')
        start_load = time.time()
        pool = Pool(2)
        self.model_ru, self.model_en = pool.map(self.load_word2vec_lambda,
                                                [model_file_ru, model_file_en])
        pool.close()
        print('loaded word2vec : time %f' % (time.time() - start_load))

        self.text_process = TextProcess(modelfile_ru=pipe_ru, modelfile_en=pipe_en)

    def load_word2vec_lambda(self, file_name):
        return gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True, limit=self.restrict_vocab)

    def n_nearest(self, vector, lang, n_limit=10):
        if lang == 'en':
            data = self.model_en.most_similar(positive=[vector], topn=n_limit, restrict_vocab=10000)
        else:
            data = self.model_ru.most_similar(positive=[vector], topn=n_limit, restrict_vocab=10000)

        return [name for name, _ in data]

    def vectorize_article_mean(self, src_file):
        try:
            article, lang = self.text_process.article_process(src_file, word_limit=self.word_limit)
            if lang == 'ru' or lang == 'en':
                vector = self.vectorize_sentence_mean(article["tagged_text"], lang)
                return vector, lang, article
        except Exception as e:
            print('vectorizing error %s' % str(e))
        return None, None, None

    def vectorize_article_mean_text(self, text):
        try:
            article, lang = self.text_process.article_process_text(text, word_limit=self.word_limit)
            if lang == 'ru' or lang == 'en':
                vector = self.vectorize_sentence_mean(article["tagged_text"], lang)
                return vector, lang, article
        except Exception as e:
            print('vectorizing error %s' % str(e))
        return None, None, None

    def vectorize_multiple_files(self, file_list, i=None, q=None):
        corpus_vecs = {"ru": [], "en": []}
        corpus_articles = {"ru": [], "en": []}
        j = 0
        for file in file_list:
            if j % 1000 == 0:
                print('procesed %.2f %%' % ((j / len(file_list))* 100))
            j += 1
            vec, lang, article = self.vectorize_article_mean(file)
            if lang is not None and vec is not None:
                corpus_vecs[lang].append(vec)
                corpus_articles[lang].append(article)
        if q is not None:
            q[i] = [corpus_vecs, corpus_articles]
        return corpus_vecs, corpus_articles

    def vectorize_multiple_files_multi(self, file_list):
        corpus_vecs = {"ru": [], "en": []}
        corpus_articles = {"ru": [], "en": []}

        threads = []
        manager = Manager()
        return_dict = manager.dict()

        for i, files in enumerate(slice_list(file_list, 8)):
            p = Process(target=self.vectorize_multiple_files, args=(files, i, return_dict,))
            p.start()
            threads.append(p)

        for t in threads:
            t.join()

        for vecs, articles in return_dict.values():
            corpus_vecs["ru"] += vecs["ru"]
            corpus_vecs["en"] += vecs["en"]

            corpus_articles["ru"] += articles["ru"]
            corpus_articles["en"] += articles["en"]

        return corpus_vecs, corpus_articles

    def vectorize_sentence_mean(self, text, lang='ru'):
        vector = np.zeros(300)
        words = .0
        if lang == 'en':
            for word in text:
                if word in self.model_en:
                    words += 1.0
                    vector += np.array(self.model_en.word_vec(word))
        else:
            for word in text:
                if word in self.model_ru:
                    words += 1.0
                    vector += np.array(self.model_ru.word_vec(word))
        if words > 0:
            vector /= words
        else:
            return None
        return vector

def main():
    vectorizer = Vectorizer(model_file_en='__data__/model_en.bin', model_file_ru='__data__/model_ru.bin',
                            pipe_en='__data__/syntagen.udpipe', pipe_ru='__data__/syntagru.model',
                            restrict_vocab=200000, word_limit=100)

    files = random.sample(list_files('/home/vova/PycharmProjects/TGmain/2703'), 400000)
    vecs, articles = vectorizer.vectorize_multiple_files_multi(files)

    with open('/home/vova/PycharmProjects/TG/__data__/temp_corp', "wb") as f:
        f.write(pickle.dumps(vecs, pickle.HIGHEST_PROTOCOL))

if __name__ == "__main__":
    main()


