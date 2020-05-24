from clustering.algos import af_algo
from clustering.algos import stream
from vectorizing import vectorization
from clustering import news_type
from sys_tools import preprocess
import time
from clustering.sqlite3threads import Sqlite3Worker
import numpy as np
import pickle
import os
from threading import Thread
from multiprocessing import Lock
from math import ceil
import datetime

models_names = {"society": "model_soci",
                  "economy": "model_econ",
                  "technology": "model_tech",
                  "sports": "model_spor",
                  "entertainment": "model_ent",
                  "science": "model_sci",
                  "other": "model_oth"}


def slice_list(list, parts=4):
    part_len = ceil(len(list) / parts)

    return [list[i:i + part_len] for i in range(0, len(list), part_len)]


class streem_wraper():
    def __init__(self, file=None):
        self.file_path = file
        if file is not None:
            try:
                self.algo = pickle.load(open(self.file_path, "rb"))
            except Exception as e:
                print(str(e))
                self.algo = stream.Birch(n_clusters=None, branching_factor=50, threshold=0.9)

    def fit(self, vectors):
        self.algo, splited = self.algo.partial_fit(np.array(vectors))
        return splited[0]

    def predict(self, vectors):
        return self.algo.predict(vectors)

    def save(self):
        pickle.dump(self.algo, open(self.file_path, "wb"))

class article_index:
    def __init__(self, vectorizer, clusterer, db_path, index_clustering_path):
        self.dir_path = index_clustering_path
        self.vectorizer = vectorizer
        self.clusterer = clusterer

        self.db = Sqlite3Worker(db_path, max_queue_size=1000)
        # self.db = sqlite3.connect(db_path, check_same_thread=False)

        self.models_ru = {}
        self.models_en = {}

        self.publishers = {}
        self.publishers_id = 0
        print(self.publishers)
        print(self.publishers_id)

        for key, value in models_names.items():
            self.models_en[key] = streem_wraper(file=os.path.join(index_clustering_path, value + '_en'))
            self.models_ru[key] = streem_wraper(file=os.path.join(index_clustering_path, value + '_ru'))

        self.models = {"ru": self.models_ru,
                       "en": self.models_en}

    def clear_db(self):
        result = self.db.execute('SELECT name from sqlite_master where type= "table"')
        for name in result:
            self.db.execute('DELETE FROM %s' % (name))

    def add_to_db(self, dict):
        self.db.execute("INSERT into %s (src, p_time, ttl, thread_id, lang, publisher) VALUES (?,?,?,?,?,?)"
                        % (dict["cluster"]),
                       (dict["src"], dict["p_time"], dict["ttl"], dict["thread_id"], dict["lang"], dict["publisher"]))

    def load_publisher_index(self):
        result = self.db.execute('SELECT id, url, count from publisher')
        self.publishers_id = result[-1][0]
        for id, url, count in result:
            self.publishers[url] = {"id": id,"count": count,"modified": 0}

    def save_publishers(self):
        for key, value in self.publishers.items():
            if value["modified"] == 1:
                self.db.execute('UPDATE publisher SET count = ? WHERE id = ?', (value["count"], value["id"],))
            elif value["modified"] == 2:
                self.db.execute('INSERT into publisher (url, count) VALUES (?, ?)', (key, value["count"]))


    def test_threading(self, file_list):
        self.algo = streem_wraper()
        threads_1 = {"ru": {}, "en": {}}
        threads_2 = {"ru": {}, "en": {}}

        corpus, articles = self.vectorizer.vectorize_multiple_files_multi(file_list)

        langs = ['ru', 'en']

        reshufles = 0

        for target_lang in langs:
            indexes = []
            start = time.time()
            for id, vec in enumerate(corpus[target_lang]):
                splited = self.algo.fit([vec])
                if not splited:
                    indexes.append(list(self.algo.predict([vec]))[0])
                else:
                    reshufles += 1
                    indexes = list(self.algo.predict(corpus[target_lang][:id]))

            print('time for clustering %.2f' % (time.time() - start))

            for i in range(len(indexes)):
                if threads_1[target_lang].get(indexes[i]) is None:
                    threads_1[target_lang][indexes[i]] = [articles[target_lang][i]["title"]]
                else:
                    threads_1[target_lang][indexes[i]].append(articles[target_lang][i]["title"])

            for key in threads_1[target_lang].keys():
                print('%d : %s' % (key, threads_1[target_lang][key]))

            print('reshufled %d times it is %.2f ' % (reshufles, reshufles/len(corpus[target_lang])))

            print('---------------------------------------------------------------------------------')

            start = time.time()
            self.algo = stream.Birch(n_clusters=None, branching_factor=50, threshold=0.9)
            self.algo.fit(np.array(corpus[target_lang]))
            indexes = self.algo.predict(np.array(corpus[target_lang]))

            for i in range(len(indexes)):
                if threads_2[target_lang].get(indexes[i]) is None:
                    threads_2[target_lang][indexes[i]] = [articles[target_lang][i]["title"]]
                else:
                    threads_2[target_lang][indexes[i]].append(articles[target_lang][i]["title"])

            for key in threads_2[target_lang].keys():
                print('%d : %s' % (key, threads_2[target_lang][key]))
            print('time for clustering %.2f' % (time.time() - start))

    def fit_models(self, vectors):
        for lang in ['ru', 'en']:
            corpus_category = {"society": [],
                               "economy": [],
                               "technology": [],
                               "sports": [],
                               "entertainment": [],
                               "science": [],
                               "other": []}
            clusters = self.clusterer.predict_multiple(vectors[lang], lang)
            for i, cluster in enumerate(clusters):
                if cluster[0] != "not_news":
                    corpus_category[cluster[0]].append(vectors[lang][i])

            for key in models_names.keys():
                self.models[lang][key].fit(corpus_category[key])

    def index_article(self, text, file_name, ttl, mutex=None):
        vector, lang, article = self.vectorizer.vectorize_article_mean_text(text)
        if vector is not None and lang is not None:
            cluster = self.clusterer.predict_single_vector(vector, lang)[0]
            if cluster != "not_news":
                thread_id = self.models[lang][cluster].predict([vector])[0]

                publisher_info = self.publishers.get(article["name"])
                if publisher_info is not None:
                    self.publishers[article["name"]]["count"] += 1
                    self.publishers[article["name"]]["modified"] = \
                        1 if self.publishers[article["name"]]["modified"] != 2 else 2
                else:
                    self.publishers_id += 1
                    self.publishers[article["name"]] = {"id": self.publishers_id, "count": 1, "modified": 2}

                p_time = datetime.datetime.fromisoformat(article["time"]).timestamp()

                self.add_to_db({"cluster": cluster,
                                "src": file_name,
                                "p_time": p_time,
                                "ttl": ttl,
                                "thread_id": int(thread_id),
                                "lang": lang,
                                "publisher": self.publishers[article["name"]]["id"]})

    def index_multi(self, files, mutex):
        for file in files:
            self.index_article(file["text"], file["name"], file["ttl"], mutex)
    
    def multi_thread_test(self, list_files):
        threads = []
        mutex = Lock()
        files_text = []
        for file in list_files:
            with open(file, "r") as f:
                files_text.append({"text": f.read(),
                                   "name": file,
                                   "ttl": 0,
                                   "p_time": 0})

        for i, files in enumerate(slice_list(files_text, 4)):
            p = Thread(target=self.index_multi, args=(files,mutex,))
            p.start()
            threads.append(p)

        for t in threads:
            t.join()

    def __del__(self):
        for key, value in self.models_ru.items():
            self.models_ru[key].save()

        for key, value in self.models_en.items():
            self.models_en[key].save()
        self.save_publishers()

        self.db.close()

def main():
    vectorizer = vectorization.Vectorizer(pipe_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagen.udpipe',
                                          model_file_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_en.bin',
                                          pipe_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagru.model',
                                          model_file_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_ru.bin',
                                          restrict_vocab=200000, word_limit=100)

    clusterer = news_type.news_categories(vectorizer,
                                          model_file_ru='/home/vova/PycharmProjects/TG/clustering/__data__/model_ru',
                                          model_file_en='/home/vova/PycharmProjects/TG/clustering/__data__/model_en')

    n_t = article_index(vectorizer=vectorizer, clusterer=clusterer,
                        db_path='/home/vova/PycharmProjects/TG/TG.db',
                        index_clustering_path='/home/vova/PycharmProjects/TG/clustering/__data__/index')

    files = preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:10]

    n_t.clear_db()

    # with open('/home/vova/PycharmProjects/TG/__data__/temp_corp', "rb") as f:
    #     corpus = pickle.loads(f.read())

    # n_t.fit_models(corpus)

    start = time.time()
    n_t.multi_thread_test(files)
    print('time for indexing 2000 articles %.2f' % (time.time() - start))
    #
    # time_sum = 0.0
    # for file in files:
    #     with open(file, "r") as f:
    #         text = f.read()
    #         start = time.time()
    #         n_t.index_article(text, file, 0)
    #         time_sum += time.time()-start
    #         print('time for articel %.2f sec' % (time.time()-start))

    #
    # n_t.test_threading(files)


if __name__ == "__main__":
    main()