from sys_tools.preprocess import merge_files
import os

dir = os.path.dirname(__file__)
word2vec_ru = os.path.join(dir, 'vectorizing', '__data__', 'model_ru.bin')
word2vec_en = os.path.join(dir, 'vectorizing', '__data__', 'model_en.bin')
pipe_ru = os.path.join(dir, 'vectorizing', '__data__', 'syntagru.model')
pipe_en = os.path.join(dir, 'vectorizing', '__data__', 'syntagen.udpipe')
temp_corp = os.path.join(dir, '__data__', 'temp_corp')
temp_corp_2 = os.path.join(dir, '__data__', 'temp_corp_2')
temp_corp_3 = os.path.join(dir, '__data__', 'temp_corp_3')
temp_corp_4 = os.path.join(dir, '__data__', 'temp_corp_4')

temp_corp_merge = os.path.join(dir, '__data__', 'temp_corp_merge')

test_set_file = os.path.join(dir, '__data__', 'test_set.json')

merge_files([temp_corp, temp_corp_2, temp_corp_3, temp_corp_4], temp_corp_merge)