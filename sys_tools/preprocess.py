from os import walk, path
from bs4 import BeautifulSoup
from json import loads, dumps
import string


def list_files(src_dir):
    f = []
    for (dirpath, dirnames, filenames) in walk(src_dir):
        for name in filenames:
            f.append(path.join(dirpath, name))

    return f


def save_object(obj, src_file):
    with open(src_file, "w") as file:
        file.write(dumps(obj))
    
        
def load_data(src_file):
    with open(src_file, "r") as file:
        return loads(file.read())