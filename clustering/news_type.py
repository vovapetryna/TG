from clustering.algos import k_mean_algo
from vectorizing import vectorization
from sys_tools import preprocess

clusters_en = {1: [0, 5, 6,14,16,17,20,21,22,25,31,33,34],
               2: [2,4,7,24,32],
               3: [10, 15],
               4: [12, 13,18,23,29],
               5: [8, 9, 27, 30],
               6: [1],
               7: [19],
               8: [3, 11, 26, 28]}
clusters_ru = {1: [0, 1, 2, 6, 9, 14, 21, 22, 23, 24, 25, 28, 30, 31, 32, 33, 34],
               2: [11, 17, 18, 20, 29],
               3: [3],
               4: [8, 13],
               5: [4, 10],
               6: [7],
               7: [27],
               8: [5, 12, 15, 26, 19, 16]}
clusters_data = {'ru': clusters_ru,
                 'en': clusters_en}

clusters_names = {1: "society",
                  2: "economy",
                  3: "technology",
                  4: "sports",
                  5: "entertainment",
                  6: "science",
                  7: "other",
                  8: "not_news"}

def predict_cluster(cluster, lang, names=False):
    ans = []
    for key, value in clusters_data[lang].items():
        if cluster in value:
            if names:
                ans.append(clusters_names[key])
            else:
                ans.append(key)
    if len(ans) == 0:
        if names:
            ans.append(clusters_names[7])
        else:
            ans.append(7)
    return ans

class news_categories:
    def __init__(self, vectorizer, model_file_ru, model_file_en):
        self.vectorizer = vectorizer

        self.cluster_algo_ru = k_mean_algo.K_Means_wrapper(src_data=model_file_ru)
        self.cluster_algo_en = k_mean_algo.K_Means_wrapper(src_data=model_file_en)

    def cluster_predict(self, vectors, lang):
        if lang == 'en':
            return self.cluster_algo_en.predict(vectors)
        elif lang == 'ru':
            return self.cluster_algo_ru.predict(vectors)

    def predict_single(self, file_src, names=True):
        vector, lang, article = self.vectorizer.vectorize_article_mean(file_src)
        if vector is not None and lang is not None:
            cluster = self.cluster_predict([vector], lang)[0]
            if cluster == 12 and lang == 'en':
                print(article["title"])
            return predict_cluster(cluster, lang, names=names)

    def predict_single_vector(self, vector, lang, names=True):
        cluster = self.cluster_predict([vector], lang)[0]
        return predict_cluster(cluster, lang, names=names)

    def predict_multiple(self, vectors, lang):
        clusters = self.cluster_predict(vectors, lang)

        return list(map(lambda cluster: predict_cluster(cluster, lang, names=True), clusters))

def main():
    vectorizer = vectorization.Vectorizer(pipe_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagen.udpipe',
                                          model_file_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_en.bin',
                                          pipe_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagru.model',
                                          model_file_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_ru.bin',
                                          restrict_vocab=200000, word_limit=100)

    n_c = news_categories(vectorizer, model_file_ru='/home/vova/PycharmProjects/TG/clustering/__data__/model_ru',
                          model_file_en='/home/vova/PycharmProjects/TG/clustering/__data__/model_en')

    files = preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:30000]

    for file in files:
        n_c.predict_single(file)


if __name__ == "__main__":
    main()
