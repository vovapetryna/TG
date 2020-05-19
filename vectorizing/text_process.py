from bs4 import BeautifulSoup
import unicodedata
import sys
from ufal.udpipe import Model, Pipeline, ProcessingError
from langdetect import detect
import time

def parse_article(text):
    soup = BeautifulSoup(text, "html.parser")

    name = soup.find("meta", property="og:site_name")
    name = name["content"] if name else ""

    time = soup.find("meta", property="article:published_time")
    time = time["content"] if time else ""

    title = soup.find("meta", property="og:title")
    title = title["content"] if title else ""

    description = soup.find("meta", property="og:description")
    description = description["content"] if description else ""

    h1 = soup.find_all("h1", text=True)
    h1_content = ""
    if h1:
        for h1i in h1:
            h1_content += " " + h1i.text.strip()

    # h2 = soup.find_all("h2")
    # h2_content = ""
    # if h2:
    #     for h2i in h2:
    #         h2_content += " " + h2i.text.strip()
    #
    # h3 = soup.find_all("h3")
    # h3_content = ""
    # if h3:
    #     for h3i in h3:
    #         h3_content += " " + h3i.text.strip()
    #
    # h4 = soup.find_all("h4")
    # h4_content = ""
    # if h4:
    #     for h4i in h4:
    #         h4_content += " " + h4i.text.strip()
    #
    # b = soup.find_all("b")
    # b_content = ""
    # if b:
    #     for bi in b:
    #         b_content += " " + bi.text.strip()

    p = soup.find_all("p")
    p_content = ""
    if p:
        for pi in p:
            p_content += " " + pi.text.strip()

    return {"name": name,
            "time": time,
            "title": title,
            "description": description,
            "h1": h1_content,
            # "h2": h2_content,
            # "h3": h3_content,
            # "h4": h4_content,
            # "b": b_content,
            "p": p_content,
            "text": (name + " " + title + " "
            + description + " " + h1_content + " "
            # + h2_content + " " + h3_content + " "
            # + h4_content + " " + b_content + " "
            + p_content)[:100]}


def process(pipeline, text, keep_pos=True, keep_punct=False):
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []

    error = ProcessingError()
    processed = pipeline.process(text, error)
    if error.occurred():
        sys.stderr.write("An error occurred when running run_udpipe: ")
        sys.stderr.write(error.message)
        sys.stderr.write("\n")
        sys.exit(1)

    content = [l for l in processed.split('\n') if not l.startswith('#')]

    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        if not lemma or not token:
            continue
        if pos in entities:
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))
        else:
            if not named:
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))

    if not keep_punct:
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos:
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    return tagged_propn


class TextProcess:
    def __init__(self, modelfile_en, modelfile_ru, keep_props=True):
        self.keep_props = keep_props

        if keep_props:
            self.model_ru = Model.load(modelfile_ru)
            self.process_pipeline_ru = Pipeline(self.model_ru, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

            self.model_en = Model.load(modelfile_en)
            self.process_pipeline_en = Pipeline(self.model_en, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

        """filter punctuation and number (delete)"""
        self.p_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
                                   or (chr(i) in '1234567890'))

    def text_clear(self, text):
        text = text.lower().translate(self.p_tbl).split(' ')
        return text

    def article_process(self, src_file, word_limit=200, limit=True):
        with open(src_file, "r") as file:
            article = parse_article(file.read())
            if len(article["text"]) > 40:
                lang = detect(article["text"])
                if lang == 'ru' or lang == 'en':
                    words = 0
                    for key, value in article.items():
                        if key != 'time':
                            if words == word_limit:
                                article[key] = []
                            else:
                                if self.keep_props:
                                    article[key] = self.tag(value, lang)
                                else:
                                    article[key] = self.text_clear(value)
                                    article[key] = [word for word in article[key] if len(word) > 1]

                                if limit:
                                    if word_limit - words - len(article[key]) < 0:
                                        article[key] = article[key][:word_limit - words]
                                        words = word_limit
                                    else:
                                        words += len(article[key])
                    return article, lang
        return [], ''

    def tag(self, text, lang="en"):
        if lang == "en":
            return process(self.process_pipeline_en, text=text.lower())
        elif lang == "ru":
            return process(self.process_pipeline_ru, text=text.lower())



def main():
    textprocess = TextProcess(keep_props=True, modelfile_en='vectorizing/__data__/syntagen.udpipe',
                              modelfile_ru='vectorizing/__data__/syntageru.model')
    # corpus = []
    # i=0
    # for __data__ in list_files('2703'):
    #     if i % 1000 == 0:
    #         print('files procesed %d' % i)
    #     i+=1
    #     corpus.append(textprocess.article_process(__data__, limit=False))
    # with open('data2703', "w") as file:
    #     file.write(dumps(corpus))
    start = time.time()
    print(textprocess.article_process('/home/vova/PycharmProjects/TGmain/2703/20200427/00/2819513471512731.html'))
    print('total time %f' % (time.time() - start))
    # print(textprocess.article_process('2703/20200427/00/256376422603845959.html'))


if __name__ == "__main__":
    main()