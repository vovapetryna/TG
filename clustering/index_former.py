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
from vectorizing.text_process import parse_article

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

        self.models_ru = {}
        self.models_en = {}

        self.publishers = {}
        self.publishers_id = 0
        self.load_publisher_index()

        self.mean_count = 1
        self.load_data_info()

        self.mutex = Lock()

        for key, value in models_names.items():
            self.models_en[key] = streem_wraper(file=os.path.join(index_clustering_path, value + '_en'))
            self.models_ru[key] = streem_wraper(file=os.path.join(index_clustering_path, value + '_ru'))

        self.models = {"ru": self.models_ru,
                       "en": self.models_en}

    def clear_db(self):
        result = self.db.execute('SELECT name from sqlite_master where type= "table"')
        for name in result:
            if name[0] != 'data_info':
                self.db.execute('DELETE FROM %s' % (name))

    def add_to_db(self, dict):
        self.db.execute("INSERT into %s (src, p_time, ttl, thread_id, lang, publisher, p_length, title)"
                        " VALUES (?,?,?,?,?,?,?,?)"
                        % (dict["cluster"]),
                       (dict["src"], dict["p_time"], dict["ttl"], dict["thread_id"],
                        dict["lang"], dict["publisher"], dict["length"], dict["title"]))

    def update_db(self, dict):
        self.db.execute('UPDATE %s SET p_time = ?, ttl = ?, title = ? WHERE src = ?' % (dict["cluster"],),
                        (dict["p_time"], dict["ttl"], dict["title"], dict["src"]))

    def load_publisher_index(self):
        result = self.db.execute('SELECT id, url, p_count from publisher')
        if len(result) > 0:
            self.publishers_id = result[-1][0]
            for id, url, count in result:
                self.publishers[url] = {"id": id,"count": count,"modified": 0}
        else:
            self.publishers_id = 0

    def load_data_info(self):
        result = self.db.execute('SELECT mean_count FROM data_info;')
        if len(result) > 0:
            self.mean_count = result[0][0]
        else:
            self.mean_count = 0

    def save_publishers(self):
        for key, value in self.publishers.items():
            if value["modified"] == 1:
                self.db.execute('UPDATE publisher SET p_count=? WHERE id=?', (value["count"], value["id"],))
            elif value["modified"] == 2:
                self.db.execute('INSERT into publisher (url, p_count) VALUES (?, ?)', (key, value["count"]))

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

    def calc_mean_length(self, files):
        length = 0
        articles = 0
        i = 0
        for file in files:
            if i % 1000 == 0:
                print('files+procesed %d' % i)
            i += 1
            vector, lang, article = self.vectorizer.vectorize_article_mean(file)
            if vector is not None and lang is not None:
                length += article["length"]
                articles += 1

        print('mean_length %.2f' % (length/articles))

    def index_article(self, text, file_name, ttl):
        sql_part = '(SELECT src, \'society\' as cluster FROM society LEFT JOIN publisher ON society.publisher=publisher.id'
        for key in models_names.keys():
            if key != 'society':
                sql_part += ' union SELECT src, \'%s\' as cluster FROM %s ' \
                            ' LEFT JOIN publisher ON %s.publisher=publisher.id' % (key, key, key)
        sql_full = 'SELECT cluster FROM ' + sql_part + ' ) as t WHERE src = \'%s\'' % (file_name)
        if_exist = self.db.execute(sql_full)

        if len(if_exist) == 0:
            vector, lang, article = self.vectorizer.vectorize_article_mean_text(text)
            if vector is not None and lang is not None:
                cluster = self.clusterer.predict_single_vector(vector, lang)[0]
                if cluster != "not_news":
                    thread_id = self.models[lang][cluster].predict([vector])[0]

                    self.mutex.acquire()
                    try:
                        publisher_info = self.publishers.get(article["name"])
                        if publisher_info is not None:
                            self.publishers[article["name"]]["count"] += 1
                            self.publishers[article["name"]]["modified"] = \
                                1 if self.publishers[article["name"]]["modified"] != 2 else 2
                        else:
                            self.publishers_id += 1
                            self.publishers[article["name"]] = {"id": self.publishers_id, "count": 1, "modified": 2}
                    finally:
                        self.mutex.release()

                    p_time = datetime.datetime.fromisoformat(article["time"]).timestamp()

                    self.mean_count += 1

                    self.add_to_db({"cluster": cluster,
                                    "src": file_name,
                                    "p_time": p_time,
                                    "ttl": ttl,
                                    "thread_id": int(thread_id),
                                    "lang": lang,
                                    "publisher": self.publishers[article["name"]]["id"],
                                    "length": article["length"],
                                    "title": article["title"]})

                    self.db.execute('delete from %s where p_time + ttl < (select max(p_time) from %s)' %
                                    (cluster, cluster,))

                    return False
        else:
            article = parse_article(text)
            p_time = datetime.datetime.fromisoformat(article["time"]).timestamp()

            self.update_db({"cluster": if_exist[0][0],
                            "src": file_name,
                            "p_time": p_time,
                            "ttl": ttl,
                            "title": article["title"]})
                
            self.db.execute('delete from %s where p_time + ttl < (select max(p_time) from %s)' %
                            (if_exist[0][0],if_exist[0][0],))

            return True

    def db_get_threads(self, period, lang, category):
        target_time = int(time.time()) - period

        if category != 'any':
            sql_part = '(SELECT src, title, thread_id, ' \
                         '(cast(p_length as float)/4503.97*0.3  + '\
                         'cast(p_count as float)/cast(%d as float)) / '\
                         '(cast(Abs(%d - p_time) as float) + 0.0001) as metric '\
                         'FROM %s LEFT JOIN publisher ON %s.publisher=publisher.id '\
                         'WHERE lang=\'%s\' ORDER BY metric DESC) as t' % (self.mean_count,
                                                                            target_time,
                                                                            category, category, lang)
            sql_full = 'SELECT src, thread_id, title FROM ' + sql_part + \
            ' WHERE metric > (SELECT metric FROM ' + sql_part + '  LIMIT 1) / 3'
            self.mutex.acquire()

            result = self.db.execute(sql_full)

            threads = []

            cluster_thread_title = []
            if len(result) > 0:
                for _, thread, title in result:
                    cluster_thread_title.append({"cluster": category, "thread": thread, "title": title})
                    threads.append(thread)
                threads = set(threads)

                sql = ''
                if len(list(threads)) > 0:
                    sql += ' SELECT '
                    sql += ' src, thread_id, \'%s\' as cluster FROM %s WHERE (' % (category, category)
                    for t in list(threads)[:-1]:
                        sql += 'thread_id = %d or ' % (t)
                    sql += 'thread_id = %d ) and lang = \'%s\' ' % (list(threads)[-1], lang,)
                sql += ' order by thread_id'

                result = self.db.execute(sql)

                result_dict = {}
                if len(result) > 0:
                    for src, thread_id, cluster in result:
                        if result_dict.get(str(thread_id) + cluster) is None:
                            result_dict[str(thread_id) + cluster] = [src]
                        else:
                            result_dict[str(thread_id) + cluster].append(src)

                answer_list = []

                max_threads = len(result_dict.keys())
                for i, item in enumerate(cluster_thread_title):
                    if i < max_threads:
                        answer_list.append({"title": item["title"],
                                            "category": item["cluster"],
                                            "articles": result_dict[str(item["thread"]) + item["cluster"]]})
                return answer_list
        else:
            sql_part = '(SELECT src, title, thread_id,\'society\' as cluster, ' \
                  '(cast(p_length as float)/4503.97*0.3  + ' \
                  'cast(p_count as float)/cast(%d as float)) / ' \
                  '(cast(Abs(%d - p_time) as float) + 0.0001) ' \
                  'as metric FROM society LEFT JOIN publisher ON society.publisher=publisher.id ' % (self.mean_count,target_time)
            for key in models_names.keys():
                if key != 'society':
                    sql_part += ' union SELECT  src, title, thread_id,\'%s\' as cluster, ' \
                           '(cast(p_length as float)/4503.97*0.3  + ' \
                           'cast(p_count as float)/cast(%d as float)) / ' \
                           '(cast(Abs(%d - p_time) as float) + 0.0001) as metric ' \
                           'FROM %s LEFT JOIN publisher ON %s.publisher=publisher.id' % (key,
                                                                                         self.mean_count,
                                                                                         target_time,
                                                                                         key, key)
            sql_part += ' WHERE lang=\'%s\' ORDER BY metric DESC) as t' % (lang,)

            sql_full = 'SELECT src, thread_id, cluster, title FROM ' + sql_part + \
                       ' WHERE metric > (SELECT metric FROM ' + sql_part + '  LIMIT 1) / 3'

            result = self.db.execute(sql_full)

            hash_thread = {"society": [],
                  "economy": [],
                  "technology": [],
                  "sports": [],
                  "entertainment": [],
                  "science": [],
                  "other": []}

            cluster_thread_title = []
            if len(result) > 0:
                for _, thread, cluster, title in result:
                    cluster_thread_title.append({"cluster": cluster, "thread": thread, "title": title})
                    hash_thread[cluster].append(thread)
                for key in hash_thread.keys():
                    hash_thread[key] = set(hash_thread[key])

                j = 0
                sql = ''
                for key, value in hash_thread.items():
                    if len(list(value)) > 0:
                        sql += ' SELECT ' if j == 0 else ' UNION SELECT'
                        sql += ' src, thread_id, \'%s\' as cluster FROM %s WHERE (' % (key, key)
                        for t in list(value)[:-1]:
                            sql += 'thread_id = %d or ' % (t)
                        sql += 'thread_id = %d ) and lang = \'%s\' ' % (list(value)[-1], lang)
                        j += 1
                sql += ' order by thread_id'

                result = self.db.execute(sql)

                result_dict = {}

                if len(result) > 0:
                    for src, thread_id, cluster in result:
                        if result_dict.get(str(thread_id) + cluster) is None:
                            result_dict[str(thread_id) + cluster] = [src]
                        else:
                            result_dict[str(thread_id) + cluster].append(src)

                answer_list = []

                max_threads = len(result_dict.keys())
                for i, item in enumerate(cluster_thread_title):
                    if i < max_threads:
                        answer_list.append({"title": item["title"],
                                            "category": item["cluster"],
                                            "articles": result_dict[str(item["thread"])+item["cluster"]]})
                return answer_list
        return None

    def db_delete(self, name):
        sql_part = '(SELECT src, url,\'society\' as cluster FROM society LEFT JOIN publisher ON society.publisher=publisher.id'
        for key in models_names.keys():
            if key != 'society':
                sql_part += ' union SELECT src, url, \'%s\' as cluster FROM %s ' \
                            ' LEFT JOIN publisher ON %s.publisher=publisher.id' % (key, key, key)
        sql_full = 'SELECT src, cluster, url FROM ' + sql_part + ' ) as t WHERE src = \'%s\'' % (name)
        
        self.mutex.acquire()
        try:
            result = self.db.execute(sql_full)
            if len(result) == 0:
                return False
            else:
                self.db.execute('DELETE FROM \'%s\' WHERE src = \'%s\'' % (result[0][1], result[0][0],))
                if self.publishers.get(result[0][2]) is not None:
                    self.publishers[result[0][2]]["count"] -= 1
                    self.publishers[result[0][2]]["modified"] = \
                                1 if self.publishers[result[0][2]]["modified"] != 2 else 2
        finally:
            self.mutex.release()
        return True

    def get_test(self):
        for i in range(1000):
            start = time.time()
            self.db_get_threads(15635452, 'en', 'any')
            print('time for 1 get %.4f' % (time.time() - start))

    def get_test_multi(self):
        threads = []

        for i in range(8):
            p = Thread(target=self.get_test)
            p.start()
            threads.append(p)

        for t in threads:
            t.join()

    def index_multi(self, files):
        for file in files:
            self.index_article(file["text"], file["name"], file["ttl"])
    
    def multi_thread_test(self, list_files):
        threads = []
        files_text = []
        for file in list_files:
            with open(file, "r") as f:
                files_text.append({"text": f.read(),
                                   "name": file,
                                   "ttl": 2592000})

        for i, files in enumerate(slice_list(files_text, 4)):
            p = Thread(target=self.index_multi, args=(files,))
            p.start()
            threads.append(p)

        for t in threads:
            t.join()

    def __del__(self):
        for key, value in self.models_ru.items():
            self.models_ru[key].save()

        for key, value in self.models_en.items():
            self.models_en[key].save()

        self.db.execute('UPDATE data_info SET mean_count = ? WHERE id=1', (self.mean_count,))
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

    files = preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:1000]

    # n_t.clear_db()

    # with open('/home/vova/PycharmProjects/TG/__data__/temp_corp', "rb") as f:
    #     corpus = pickle.loads(f.read())

    # n_t.fit_models(corpus)

    start = time.time()
    n_t.multi_thread_test(files)
    # n_t.get_test_multi()
    # print(n_t.db_get_threads(12352342325, 'ru', 'sports'))
    # for file in files:
    #     n_t.db_delete(file)
    print('time for indexing 200 articles %.2f' % (time.time() - start))
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