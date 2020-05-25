from langdetect import detect
from vectorizing.text_process import parse_article
from multiprocessing import Process, Lock, Manager
from math import ceil
from sys_tools.preprocess import list_files
import time
import ntpath

def slice_list(list, parts=4):
    part_len = ceil(len(list) / parts)

    return [list[i:i + part_len] for i in range(0, len(list), part_len)]

class lang_detect:
    def __init__(self):
        self.mutex = Lock()

    def detect_single_file(self, file):
        with open(file, "r") as f:
            article = parse_article(f.read())

            self.mutex.acquire()
            try:
                lang = detect(article["text"])
            finally:
                self.mutex.release()

            if lang == 'ru' or lang == 'en':
                return lang

        return None

    def detect_multiple(self, file_list, i, return_dict):
        langs = {"ru": [], "en": []}
        for file in file_list:
            lang = self.detect_single_file(file)
            if lang is not None:
                langs[lang].append(ntpath.basename(file))

        return_dict[i] = langs

    def detect_multiple_threads(self, file_list):
        langs = {"ru": [], "en": []}
        threads = []
        manager = Manager()
        return_dict = manager.dict()

        for i, files in enumerate(slice_list(file_list, 8)):
            p = Process(target=self.detect_multiple, args=(files, i, return_dict,))
            p.start()
            threads.append(p)

        for t in threads:
            t.join()

        for langs_thread in return_dict.values():
            langs["ru"] += langs_thread["ru"]
            langs["en"] += langs_thread["en"]

        return [{"lang_code": "en", "articles": langs["en"]},
                {"lang_code": "ru", "articles": langs["ru"]}]

def main():
    detector = lang_detect()

    files = list_files('/home/vova/PycharmProjects/TGmain/2703')[:1000]

    start = time.time()
    print(detector.detect_multiple_threads(files))
    print('detection time %.2f' % (time.time() - start))

if __name__ == "__main__":
    main()


