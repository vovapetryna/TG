import gensim
import time
import numpy as np
from vectorizing.text_process import TextProcess

sum_weights = {"name": 0.1,
               "title": 0.15,
               "description": 0.2,
               "h1": 0.15,
               "h2": 0.05,
               "h3": 0.05,
               "h4": 0.05,
               "b": 0.1,
               "p": 0.15}

sum_weights = {"name": 0.1,
               "title": 0.1,
               "description": 0.25,
               "h1": 0.25,
               "p": 0.3}



class Vectorizer:
    def __init__(self, model_file_ru, model_file_en, pipe_ru, pipe_en):
        print('vectorizer start loading')
        start_load = time.time()

        self.model_ru = gensim.models.KeyedVectors.load_word2vec_format(model_file_ru, binary=True)
        print('loaded RU word2vec : time %f' % (time.time() - start_load))

        self.model_en = gensim.models.KeyedVectors.load_word2vec_format(model_file_en, binary=True)
        print('loaded EN word2vec : time %f' % (time.time() - start_load))

        self.text_process = TextProcess(modelfile_ru=pipe_ru, modelfile_en=pipe_en)
        print('loaded pipelines : time %f' % (time.time() - start_load))

        print('models loaded with %f sec' % (time.time() - start_load))

    def n_nearest(self, vector, lang, n_limit = 10):
        if lang == 'en':
            data = self.model_en.most_similar(positive=[vector], topn=n_limit, restrict_vocab=10000)
        else:
            data = self.model_ru.most_similar(positive=[vector], topn=n_limit, restrict_vocab=10000)

        return [name for name, _ in data]

    def vectorize_article_mean(self, src_file, word_limit=200):
        try:
            article, lang = self.text_process.article_process(src_file, word_limit=word_limit, limit=True)
            if lang == 'ru' or lang == 'en':
                vector = np.zeros(300)
                vector_c = 0.0
                for part, weight in sum_weights.items():
                    if len(article[part]) > 0:
                        vector += self.vectorize_sentence_mean(article[part], lang)

                if vector_c > 0:
                    vector /= vector_c
                return vector, lang
        except Exception as e:
            print(str(e))

        return None, None

    def vectorize_article(self, src_file, word_limit=200):
        article, lang = self.text_process.article_process(src_file, word_limit=word_limit, limit=True)

        if lang == 'ru' or lang == 'en':
            vector = np.zeros(300)

            for part, weight in sum_weights.items():
                if len(article[part]) > 0:
                    vector += weight * self.vectorize_sentence_mean(article[part], lang)

            return vector, lang
        else:
            return None, None

    def vectorize_sentence_mean(self, text, lang='ru'):
        vector = np.zeros(300)
        words = .0
        loss = .0
        if lang == 'en':
            for word in text:
                if word in self.model_en:
                    words += 1.0
                    vector += np.array(self.model_en.word_vec(word))
                else:
                    loss += 1.0
        else:
            for word in text:
                if word in self.model_ru:
                    words += 1.0
                    vector += np.array(self.model_ru.word_vec(word))
                else:
                    loss += 1.0
        if words > 0:
            vector /= words
        # print('info loss factor %.2f' % (loss / len(text)))
        return vector


def main():
    vectorizer = Vectorizer(model_file_en='__data__/model_en.bin', model_file_ru='__data__/model_ru.bin',
                            pipe_en='__data__/syntagen.udpipe', pipe_ru='__data__/syntagru.model')

    start = time.time()
    # vector, _ = vectorizer.vectorize_article_mean('/home/vova/PycharmProjects/TGmain/2703/20200427/01/993065328833743.html',
    #                                               word_limit=100)
    # print(vectorizer.n_nearest(vector, 'en'))
    # vector, _ = vectorizer.vectorize_article_mean('/home/vova/PycharmProjects/TGmain/2703/20200427/01/993066831009800.html',
    #                                               word_limit=100)
    # print(vectorizer.n_nearest(vector, 'en'))
    # vector, _ = vectorizer.vectorize_article_mean('/home/vova/PycharmProjects/TGmain/2703/20200427/01/619043747613953010.html',
    #                                               word_limit=100)
    # print(vectorizer.n_nearest(vector, 'ru'))
    # vector, _ = vectorizer.vectorize_article_mean('/home/vova/PycharmProjects/TGmain/2703/20200427/01/6450858316899294513.html',
    #                                               word_limit=100)
    # print(vectorizer.n_nearest(vector, 'en'))
    # vector, _ = vectorizer.vectorize_article_mean('/home/vova/PycharmProjects/TGmain/2703/20200427/01/6679535024845809554.html',
    #                                               word_limit=100)
    # print(vectorizer.n_nearest(vector, 'en'))
    # print('time for vectorizing %.2f' % ((time.time() - start) / 3))

    vector = np.array(vectorizer.model_ru.word_vec('политика_NOUN'))
    vector += np.array(vectorizer.model_ru.word_vec('ложь_NOUN'))
    print(vectorizer.n_nearest(vector / 3, 'ru'))


if __name__ == "__main__":
    main()
