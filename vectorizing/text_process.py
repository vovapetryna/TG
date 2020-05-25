from bs4 import BeautifulSoup
import unicodedata
import sys
from ufal.udpipe import Model, Pipeline
from langdetect import detect
import time
from sys_tools import preprocess
from collections import Counter
from json import loads
from threading import Lock

mutex = Lock()

en_stop_words = loads('{"words":["able","about","above","abroad","according","accordingly","across","actually","adj","after","afterwards","again","against","ago","ahead","ain\u0027t","all","allow","allows","almost","alone","along","alongside","already","also","although","always","am","amid","amidst","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren\u0027t","around","as","a\u0027s","aside","ask","asking","associated","at","available","away","awfully","back","backward","backwards","be","became","because","become","becomes","becoming","been","before","beforehand","begin","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","came","can","cannot","cant","can\u0027t","caption","cause","causes","certain","certainly","changes","clearly","c\u0027mon","co","co.","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn\u0027t","course","c\u0027s","currently","dare","daren\u0027t","definitely","described","despite","did","didn\u0027t","different","directly","do","does","doesn\u0027t","doing","done","don\u0027t","down","downwards","during","each","edu","eg","eight","eighty","either","else","elsewhere","end","ending","enough","entirely","especially","et","etc","even","ever","evermore","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","fairly","far","farther","few","fewer","fifth","first","five","followed","following","follows","for","forever","former","formerly","forth","forward","found","four","from","further","furthermore","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn\u0027t","half","happens","hardly","has","hasn\u0027t","have","haven\u0027t","having","he","he\u0027d","he\u0027ll","hello","help","hence","her","here","hereafter","hereby","herein","here\u0027s","hereupon","hers","herself","he\u0027s","hi","him","himself","his","hither","hopefully","how","howbeit","however","hundred","i\u0027d","ie","if","ignored","i\u0027ll","i\u0027m","immediate","in","inasmuch","inc","inc.","indeed","indicate","indicated","indicates","inner","inside","insofar","instead","into","inward","is","isn\u0027t","it","it\u0027d","it\u0027ll","its","it\u0027s","itself","i\u0027ve","just","k","keep","keeps","kept","know","known","knows","last","lately","later","latter","latterly","least","less","lest","let","let\u0027s","like","liked","likely","likewise","little","look","looking","looks","low","lower","ltd","made","mainly","make","makes","many","may","maybe","mayn\u0027t","me","mean","meantime","meanwhile","merely","might","mightn\u0027t","mine","minus","miss","more","moreover","most","mostly","mr","mrs","much","must","mustn\u0027t","my","myself","name","namely","nd","near","nearly","necessary","need","needn\u0027t","needs","neither","never","neverf","neverless","nevertheless","new","next","nine","ninety","no","nobody","non","none","nonetheless","noone","no-one","nor","normally","not","nothing","notwithstanding","novel","now","nowhere","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","one\u0027s","only","onto","opposite","or","other","others","otherwise","ought","oughtn\u0027t","our","ours","ourselves","out","outside","over","overall","own","particular","particularly","past","per","perhaps","placed","please","plus","possible","presumably","probably","provided","provides","que","quite","qv","rather","rd","re","really","reasonably","recent","recently","regarding","regardless","regards","relatively","respectively","right","round","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","shan\u0027t","she","she\u0027d","she\u0027ll","she\u0027s","should","shouldn\u0027t","since","six","so","some","somebody","someday","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that\u0027ll","thats","that\u0027s","that\u0027ve","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","there\u0027d","therefore","therein","there\u0027ll","there\u0027re","theres","there\u0027s","thereupon","there\u0027ve","these","they","they\u0027d","they\u0027ll","they\u0027re","they\u0027ve","thing","things","think","third","thirty","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","till","to","together","too","took","toward","towards","tried","tries","truly","try","trying","t\u0027s","twice","two","un","under","underneath","undoing","unfortunately","unless","unlike","unlikely","until","unto","up","upon","upwards","us","use","used","useful","uses","using","usually","v","value","various","versus","very","via","viz","vs","want","wants","was","wasn\u0027t","way","we","we\u0027d","welcome","well","we\u0027ll","went","were","we\u0027re","weren\u0027t","we\u0027ve","what","whatever","what\u0027ll","what\u0027s","what\u0027ve","when","whence","whenever","where","whereafter","whereas","whereby","wherein","where\u0027s","whereupon","wherever","whether","which","whichever","while","whilst","whither","who","who\u0027d","whoever","whole","who\u0027ll","whom","whomever","who\u0027s","whose","why","will","willing","wish","with","within","without","wonder","won\u0027t","would","wouldn\u0027t","yes","yet","you","you\u0027d","you\u0027ll","your","you\u0027re","yours","yourself","yourselves","you\u0027ve","zero","a","how\u0027s","i","when\u0027s","why\u0027s","b","c","d","e","f","g","h","j","l","m","n","o","p","q","r","s","t","u","uucp","w","x","y","z","I","www","amoungst","amount","bill","bottom","call","computer","con","couldnt","cry","de","describe","detail","due","eleven","empty","fifteen","fify","fill","find","fire","forty","front","full","give","hasnt","herse”","himse”","interest","itse”","mill","move","myse”","part","put","show","side","sincere","sixty","system","ten","thick","thin","top","twelve","twenty","abst","accordance","act","added","adopted","affected","affecting","affects","ah","announce","anymore","apparently","approximately","aren","arent","arise","auth","beginning","beginnings","begins","biol","briefly","ca","date","ed","effect","et-al","ff","fix","gave","giving","hed","heres","hes","hid","home","id","im","immediately","importance","important","index","information","invention","itd","keys","kg","km","largely","lets","line","\u0027ll","means","mg","million","ml","mug","na","nay","necessarily","nos","noted","obtain","obtained","omitted","ord","owing","page","pages","poorly","possibly","potentially","pp","predominantly","present","previously","primarily","promptly","proud","quickly","ran","readily","ref","refs","related","research","resulted","resulting","results","run","sec","section","shed","shes","showed","shown","showns","shows","significant","significantly","similar","similarly","slightly","somethan","specifically","state","states","stop","strongly","substantially","successfully","sufficiently","suggest","thered","thereof","therere","thereto","theyd","theyre","thou","thoughh","thousand","throug","til","tip","ts","ups","usefully","usefulness","\u0027ve","vol","vols","wed","whats","wheres","whim","whod","whos","widely","words","world","youd","youre"]}')
ru_stop_words = loads('{"words":["а","е","и","ж","м","о","на","не","ни","об","но","он","мне","мои","мож","она","они","оно","мной","много","многочисленное","многочисленная","многочисленные","многочисленный","мною","мой","мог","могут","можно","может","можхо","мор","моя","моё","мочь","над","нее","оба","нам","нем","нами","ними","мимо","немного","одной","одного","менее","однажды","однако","меня","нему","меньше","ней","наверху","него","ниже","мало","надо","один","одиннадцать","одиннадцатый","назад","наиболее","недавно","миллионов","недалеко","между","низко","меля","нельзя","нибудь","непрерывно","наконец","никогда","никуда","нас","наш","нет","нею","неё","них","мира","наша","наше","наши","ничего","начала","нередко","несколько","обычно","опять","около","мы","ну","нх","от","отовсюду","особенно","нужно","очень","отсюда","в","во","вон","вниз","внизу","вокруг","вот","восемнадцать","восемнадцатый","восемь","восьмой","вверх","вам","вами","важное","важная","важные","важный","вдали","везде","ведь","вас","ваш","ваша","ваше","ваши","впрочем","весь","вдруг","вы","все","второй","всем","всеми","времени","время","всему","всего","всегда","всех","всею","всю","вся","всё","всюду","г","год","говорил","говорит","года","году","где","да","ее","за","из","ли","же","им","до","по","ими","под","иногда","довольно","именно","долго","позже","более","должно","пожалуйста","значит","иметь","больше","пока","ему","имя","пор","пора","потом","потому","после","почему","почти","посреди","ей","два","две","двенадцать","двенадцатый","двадцать","двадцатый","двух","его","дел","или","без","день","занят","занята","занято","заняты","действительно","давно","девятнадцать","девятнадцатый","девять","девятый","даже","алло","жизнь","далеко","близко","здесь","дальше","для","лет","зато","даром","первый","перед","затем","зачем","лишь","десять","десятый","ею","её","их","бы","еще","при","был","про","процентов","против","просто","бывает","бывь","если","люди","была","были","было","будем","будет","будете","будешь","прекрасно","буду","будь","будто","будут","ещё","пятнадцать","пятнадцатый","друго","другое","другой","другие","другая","других","есть","пять","быть","лучше","пятый","к","ком","конечно","кому","кого","когда","которой","которого","которая","которые","который","которых","кем","каждое","каждая","каждые","каждый","кажется","как","какой","какая","кто","кроме","куда","кругом","с","т","у","я","та","те","уж","со","то","том","снова","тому","совсем","того","тогда","тоже","собой","тобой","собою","тобою","сначала","только","уметь","тот","тою","хорошо","хотеть","хочешь","хоть","хотя","свое","свои","твой","своей","своего","своих","свою","твоя","твоё","раз","уже","сам","там","тем","чем","сама","сами","теми","само","рано","самом","самому","самой","самого","семнадцать","семнадцатый","самим","самими","самих","саму","семь","чему","раньше","сейчас","чего","сегодня","себе","тебе","сеаой","человек","разве","теперь","себя","тебя","седьмой","спасибо","слишком","так","такое","такой","такие","также","такая","сих","тех","чаще","четвертый","через","часто","шестой","шестнадцать","шестнадцатый","шесть","четыре","четырнадцать","четырнадцатый","сколько","сказал","сказала","сказать","ту","ты","три","эта","эти","что","это","чтоб","этом","этому","этой","этого","чтобы","этот","стал","туда","этим","этими","рядом","тринадцать","тринадцатый","этих","третий","тут","эту","суть","чуть","тысяч"]}')

def parse_article(text):
    soup = BeautifulSoup(text, "html.parser")

    url = soup.find("meta", property="og:url")
    url = url["content"] if url else ""
    
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

    p = soup.find_all("p")
    p_content = ""
    if p:
        for pi in p:
            p_content += " " + pi.text.strip()

    return {"length": len(text),
            "name": name,
            "time": time,
            "title": title,
            "description": description,
            "h1": h1_content,
            "p": p_content,
            "text": (name + " " + title + " "
            + description + " " + h1_content + " "
            + p_content)[:600]}


def process(pipeline, text, keep_pos=True, keep_punct=False, lang='en'):
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []

    processed = pipeline.process(text)

    content = [l for l in processed.split('\n') if not l.startswith('#')]

    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        if lang == 'ru' and token in ru_stop_words["words"]:
            continue
        if lang == 'en' and token in en_stop_words["words"]:
            continue
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

def term_frequency(list):
    size = len(list)
    tf = Counter(list)
    for w in tf.keys():
        tf[w] /= size
    return tf

class TextProcess:
    def __init__(self, modelfile_en, modelfile_ru, keep_props=True):
        self.keep_props = keep_props

        if keep_props:
            start_load = time.time()
            self.model_ru = Model.load(modelfile_ru)
            self.process_pipeline_ru = Pipeline(self.model_ru, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

            self.model_en = Model.load(modelfile_en)
            self.process_pipeline_en = Pipeline(self.model_en, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

            print('loaded piplines : time %f' % (time.time() - start_load))

        """filter punctuation and number (delete)"""
        self.p_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')
                                   or (chr(i) in '1234567890'))


    def text_clear(self, text):
        text = text.lower().translate(self.p_tbl).split(' ')
        return text

    def gen_tagged_corpus(self, file_list, word_limit=100):
        corpus = []
        for file in file_list:
            if len(corpus) == 10000:
                break
            article, lang = self.article_process(file, word_limit=word_limit)
            sentance = []
            if article != [] and lang != '':
                for key, value in article.items():
                    if key != 'time':
                        sentance += value
                corpus.append(sentance)
        return corpus

    def article_process_text(self, text, word_limit=200):
        article = parse_article(text)

        if len(article["text"]) > 40:
            mutex.acquire()
            try:
                lang = detect(article["text"][:100])
            finally:
                mutex.release()
            if lang == 'ru' or lang == 'en':
                article["tagged_text"] = self.tag(article["text"], lang)[:word_limit]
                article["TF"] = term_frequency(article["tagged_text"])
                # print(article["TF"])
                if article["tagged_text"] == []:
                    return [], ''
                return article, lang

        return [], ''

    def article_process(self, src_file, word_limit=200):
        with open(src_file, "r") as file:
            file_data = file.read()
            article = parse_article(file_data)

            if len(article["text"]) > 40:
                mutex.acquire()
                try:
                    lang = detect(article["text"][:100])
                finally:
                    mutex.release()
                if lang == 'ru' or lang == 'en':
                    article["tagged_text"] = self.tag(article["text"], lang)[:word_limit]
                    # article["TF"] = term_frequency(article["tagged_text"])
                    # print(article["TF"])
                    if article["tagged_text"] == []:
                        return [], ''
                    return article, lang
        return [], ''

    def tag(self, text, lang):
        if lang == "en":
            return process(self.process_pipeline_en, text=text.lower(), lang=lang)
        elif lang == "ru":
            return process(self.process_pipeline_ru, text=text.lower(), lang=lang)



def main():
    textprocess = TextProcess(keep_props=True, modelfile_en='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntagen.udpipe',
                              modelfile_ru='/home/vova/PycharmProjects/TG/vectorizing/__data__/syntageru.model')

    corp = textprocess.gen_tagged_corpus(preprocess.list_files('/home/vova/PycharmProjects/TGmain/2703')[:10])
    preprocess.save_object(corp, '/home/vova/PycharmProjects/TG/__data__/text_corp')


if __name__ == "__main__":
    main()