# Get request
# importing the requests library
import requests

def make_get_request(port, period,lang_code, category):
    url = "http://localhost:{}/threads".format(port)
    PARAMS = {'period': period,
              'lang_code' : lang_code,
              'category': category}
    return requests.get(url=url, params=PARAMS).json()

def make_put_request(port, article, ttl, data):
    url = "http://localhost:{}/{}".format(port, article)
    headers = {'Cache-Control': "max-age=".format(ttl)}
    return requests.put(url=url, headers=headers, data=data).json()

def make_pu_request(port, article):
    url = "http://localhost:{}/{}".format(port, article)
    return requests.delete(url=url).json()

