from clustering.algos import k_mean_algo

clusters_en = {4: [11, 10, 3], 1: [7, 3, 4, 6, 2, 1, 8], 5: [0, 12], 2: [5, 4, 15], 6: [8], 3: [15, 13], 8: [12, 8]}
clusters_ru = {1: [0, 14, 10, 4, 3, 12, 7, 5, 15, 6, 11], 2: [3], 5: [14, 9], 4: [8], 8: [1], 7: [2], 3: [13]}
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

    def predict_single(self, file_src):
        vector, lang, _ = self.vectorizer.vectorize_article_mean(file_src)

        cluster = self.cluster_predict([vector], lang)[0]
        return predict_cluster(cluster, lang, names=True)

    def predict_multiple(self, vectors, lang):
        clusters = self.cluster_predict(vectors, lang)

        return map(lambda cluster: predict_cluster(cluster, lang, names=True), clusters)
