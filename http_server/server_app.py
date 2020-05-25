# server_simple.py

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from threading import Timer
from furl import furl
from socketserver import ThreadingMixIn
import time
import json

from clustering import index_former

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

class HandlerClass(BaseHTTPRequestHandler):
    def _set_response(self, vectorizer, clusterer, db_path, index_clustering_path):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.indexer = index_former.article_index(vectorizer=vectorizer,
                                                  clusterer=clusterer,
                                                  db_path=db_path,
                                                  index_clustering_path=index_clustering_path)


    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))

        query = furl(self.path)

        # reading params from request
        period = int(query.args['period'])
        lang_code = query.args['lang_code']
        category = query.args['category']
    
        # replace by deep copy?
        filtered_data = json.dumps(self.indexer.db_get_threads(period, lang_code, category))

        print("{} | {} | {}".format(period, lang_code, category))
        # apply filters <period>
        # apply filters <lang_code>
        # apply filters <category>
        
        self._set_response()
        # valid json format expected
        self.wfile.write(str(filtered_data).encode('utf-8'))

    def do_PUT(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        index = self.path.split('?')[0]
        is_replaced = False

        time_to_live = int(self.headers['Cache-Control'].split('=')[1])
        text = str(self.rfile.read(content_length).decode('utf-8'))

        is_replaced = self.indexer.index_article(text, index, time_to_live)

        #response to client
        self._set_response()

        if is_replaced:
            self.wfile.write("HTTP/1.1  204 Updated".encode('utf-8'))
        else:
            self.wfile.write("HTTP/1.1  201 Created".encode('utf-8'))

    def do_DELETE(self):
        index = self.path.split('?')[0]
        result = self.indexer.db_delete(index)
        self._set_response()
        if result:
            self.wfile.write("HTTP/1.1 204 No Content".encode('utf-8'))
        else:
            self.wfile.write("HTTP/1.1 404 Item Does not exist".encode('utf-8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def init_server(host = '', port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, HandlerClass)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')