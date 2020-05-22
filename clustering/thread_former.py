from clustering.algos import af_algo
from clustering.algos import agglomerative_algo
from vectorizing import vectorization
from sys_tools import preprocess
import time

class news_thred:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

        # self.algo = af_algo.AffinityPropagation_algo_wrapper()
        self.algo_ru = agglomerative_algo.AgglomerativeClustering_algo_wrapper(scale=0.12)
        self.algo_en = agglomerative_algo.AgglomerativeClustering_algo_wrapper(scale=0.2)

    def form_thread(self, file_list):
        threads = {"ru": {}, "en": {}}

        corpus, articles = self.vectorizer.vectorize_multiple_files(file_list)

        self.algo_ru.fit(corpus["ru"])
        indexes = self.algo_ru.predict(corpus["ru"])

        for i in range(len(indexes)):
            if threads['ru'].get(indexes[i]) is None:
                threads['ru'][indexes[i]] = [articles['ru'][i]["title"]]
            else:
                threads['ru'][indexes[i]].append(articles['ru'][i]["title"])

        self.algo_en.fit(corpus["en"])
        indexes = self.algo_en.predict(corpus["en"])

        for i in range(len(indexes)):
            if threads['en'].get(indexes[i]) is None:
                threads['en'][indexes[i]] = [articles['en'][i]["title"]]
            else:
                threads['en'][indexes[i]].append(articles['en'][i]["title"])

        for key in threads['ru'].keys():
            print(threads['ru'][key])

        for key in threads['en'].keys():
            print(threads['en'][key])

        for key, value in threads['en'].items():
            print(str(key) + ' ' + str(len(value)))


def main():
    vectorizer = vectorization.Vectorizer(pipe_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagen.udpipe',
                                          model_file_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_en.bin',
                                          pipe_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagru.model',
                                          model_file_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_ru.bin',
                                          restrict_vocab=200000, word_limit=100)
    n_t = news_thred(vectorizer=vectorizer)

    start = time.time()
    files = preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:5000]

    n_t.form_thread(files)

    print('time for threading %.2f' % (time.time() - start))


if __name__ == "__main__":
    main()