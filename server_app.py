# server_simple.py

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from threading import Timer
from furl import furl
from socketserver import ThreadingMixIn
import time
import json
import os

from clustering import index_former
from clustering import news_type
from vectorizing.vectorization import Vectorizer

#
# Entities = {}
#
# def remove_element(name_id):
#     print("Called deleter {}".format(name_id))
#
#     is_exist_in_dictionary = Entities.get(name_id) is not None
#
#     if is_exist_in_dictionary:
#         print("actually deleted")
#         Entities.pop(name_id)
#
#     return is_exist_in_dictionary

def init_vectorizer():
    dir = os.path.dirname(__file__)
    word2vec_ru = os.path.join(dir, '__data__', 'model_ru.bin')
    word2vec_en = os.path.join(dir, '__data__', 'model_en.bin')
    pipe_ru = os.path.join(dir, '__data__', 'syntagru.model')
    pipe_en = os.path.join(dir, '__data__', 'syntagen.udpipe')

    return Vectorizer(model_file_en=word2vec_en, model_file_ru=word2vec_ru,
                                pipe_en=pipe_en, pipe_ru=pipe_ru, restrict_vocab=200000, word_limit=100)

def init_clusterer(vectorizer):
    dir = os.path.dirname(__file__)
    model_en = os.path.join(dir, '__data__', 'model_en')
    model_ru = os.path.join(dir, '__data__', 'model_ru')

    return news_type.news_categories(vectorizer, model_en, model_ru)

class HandlerClass(BaseHTTPRequestHandler):

    def do_GET(self):
        if is_db_loaded:
            self.wfile.write("HTTP/1.1 503 Service Unavailable")

        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))

        query = furl(self.path)

        # reading params from request
        period = int(query.args['period'])
        lang_code = query.args['lang_code']
        category = query.args['category']
    
        # replace by deep copy?
        global indexer
        filtered_data = json.dumps(indexer.db_get_threads(period, lang_code, category))

        # apply filters <period>
        # apply filters <lang_code>
        # apply filters <category>

        self.wfile.write(str(filtered_data).encode('utf-8'))

    def do_PUT(self):
        if is_db_loaded:
            self.wfile.write("HTTP/1.1 503 Service Unavailable")

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        index = self.path.split('?')[0]
        is_replaced = False

        time_to_live = int(self.headers['Cache-Control'].split('=')[1])
        text = str(self.rfile.read(content_length).decode('utf-8'))
        global indexer
        is_replaced = indexer.index_article(self=indexer, text=text, file_name=index, ttl=time_to_live)

        if is_replaced:
            self.wfile.write("HTTP/1.1  204 Updated".encode('utf-8'))
        else:
            self.wfile.write("HTTP/1.1  201 Created".encode('utf-8'))

    def do_DELETE(self):
        if is_db_loaded:
            self.wfile.write("HTTP/1.1 503 Service Unavailable")

        index = self.path.split('?')[0]
        global indexer
        result = indexer.db_delete(index)

        if result:
            self.wfile.write("HTTP/1.1 204 No Content".encode('utf-8'))
        else:
            self.wfile.write("HTTP/1.1 404 Item Does not exist".encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

indexer = index_former.article_index
is_db_loaded = False

def init_server(host = '', port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)

    httpd = ThreadedHTTPServer(server_address, HandlerClass)

    vectorizer = init_vectorizer()
    clusterer = init_clusterer(vectorizer)

    dir = os.path.dirname(__file__)

    db_path = os.path.join(dir, 'TG.db')
    index_clustering_path = os.path.join(dir, '__data__', 'index')

    indexer = index_former.article_index(vectorizer=vectorizer,
                                         clusterer=clusterer,
                                         db_path=db_path,
                                         index_clustering_path=index_clustering_path)
    is_db_loaded = True

    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')