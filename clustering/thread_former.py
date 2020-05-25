from clustering.algos import agglomerative_algo
from vectorizing import vectorization
from sys_tools import preprocess
import time

class news_thred:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.algo = agglomerative_algo.AgglomerativeClustering_algo_wrapper(scale=0.15)

    def form_thread(self, file_list):
        threads = {"ru": {}, "en": {}}
        threads_names = {"ru": {}, "en": {}}

        corpus, articles = self.vectorizer.vectorize_multiple_files_multi(file_list)
        langs = ['ru', 'en']

        for target_lang in langs:
            self.algo.fit(corpus[target_lang])
            indexes = self.algo.predict(corpus[target_lang])

            for i in range(len(indexes)):
                if threads[target_lang].get(indexes[i]) is None:
                    threads[target_lang][indexes[i]] = [articles[target_lang][i]["file_name"]]
                    if len(articles[target_lang][i]["title"]) > 60:
                        threads_names[target_lang][indexes[i]] = articles[target_lang][i]["title"][:60] + "..."
                    else:
                        threads_names[target_lang][indexes[i]] = articles[target_lang][i]["title"]
                else:
                    threads[target_lang][indexes[i]].append(articles[target_lang][i]["file_name"])

            ans = []

            for key in threads[target_lang].keys():
                ans.append({"title": threads_names[target_lang][key], "articles": threads[target_lang][key]})
        return ans

def main():
    vectorizer = vectorization.Vectorizer(pipe_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagen.udpipe',
                                          model_file_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_en.bin',
                                          pipe_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagru.model',
                                          model_file_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/model_ru.bin',
                                          restrict_vocab=200000, word_limit=100)
    n_t = news_thred(vectorizer=vectorizer)

    start = time.time()
    files = preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:1000]
    print(n_t.form_thread(files))
    print('time for threading %.2f' % (time.time() - start))


if __name__ == "__main__":
    main()