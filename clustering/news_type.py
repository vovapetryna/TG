from clustering.algos import k_mean_algo

clusters_en = {}
clusters_ru = {1: [2, 18, 5, 1, 13, 23, 14, 9, 27, 0, 6, 20, 4, 15, 28, 7, 29, 26, 11, 19, 24, 16], 2: [13, 19], 5: [8, 18], 4: [3], 8: [10], 6: [11], 7: [17, 4, 9, 21], 3: [12]}
clusters_names = {1: "society",
                  2: "economy",
                  3: "technology",
                  4: "sports",
                  5: "entertainment",
                  6: "science",
                  7: "other",
                  8: "not_news"}

class news_categories:
    def __init__(self, vectorizer, model_file_ru, model_file_en):
        self.vectorizer = vectorizer

        self.cluster_algo_ru = k_mean_algo.K_Means_wrapper(src_data=model_file_ru)
        self.cluster_algo_en = None