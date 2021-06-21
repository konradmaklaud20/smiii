import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from fake_headers import Headers
from simpletransformers.seq2seq import Seq2SeqModel
import datetime
import re
import spacy
import random
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import nltk
import pickle
import keras
import numpy as np
import string
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


class Size:
    def __init__(self):
        self.large = 'large'
        self.medium = 'medium'
        self.small = 'small'


class Preview:
    def __init__(self, url: str,
                 width: int,
                 height: int):
        self.url = url
        self.width = width
        self.height = height
        self.size = str(width) + '*' + str(height)


class Result1:
    def __init__(self, title: (str, None),
                 description: (str, None),
                 domain: str,
                 url: str,
                 width: int,
                 height: int,
                 preview: Preview):
        self.title = title
        self.description = description
        self.domain = domain
        self.url = url
        self.width = width
        self.height = height
        self.size = str(width) + '*' + str(height)
        self.preview = preview


class YandexImage1:
    def __init__(self):
        self.size = Size()
        self.headers = Headers(headers=True).generate()
        self.version = '1.0-release'
        self.about = 'Yandex Images Parser'

    def search(self, query: str, sizes: Size = 'large') -> list:
        request = requests.get('https://yandex.ru/images/search',
                               params={"text": query,
                                       "nomisspell": 1,
                                       "noreask": 1,
                                       "isize": sizes
                                       },
                               headers=self.headers)

        soup = BeautifulSoup(request.text, 'html.parser')
        items_place = soup.find('div', {"class": "serp-list"})
        output = list()
        try:
            items = items_place.find_all("div", {"class": "serp-item"})
        except AttributeError as e:
            print(e)
            return output

        for item in items:
            data = json.loads(item.get("data-bem"))
            image = data['serp-item']['img_href']
            image_width = data['serp-item']['preview'][0]['w']
            image_height = data['serp-item']['preview'][0]['h']

            snippet = data['serp-item']['snippet']
            try:
                title = snippet['title']
            except KeyError:
                title = None
            try:
                description = snippet['text']
            except KeyError:
                description = None
            domain = snippet['domain']

            preview = 'https:' + data['serp-item']['thumb']['url']
            preview_width = data['serp-item']['thumb']['size']['width']
            preview_height = data['serp-item']['thumb']['size']['height']

            output.append(Result1(title, description, domain, image,
                                 image_width, image_height,
                                 Preview(preview, preview_width, preview_height)))

        return output


class Result:
    def __init__(self,
                 url: str
                 ):
        self.url = url


class YandexImage:
    def search(self, text):
        r = requests.get('https://api.proxyscrape.com/shareget.php?id=l5ty3b9')
        r2 = requests.get(r.text)
        proxi = random.choice(["http://" + i for i in
                               r2.text.replace('\r', '').split('\n') if len(i) > 1])
        headers = Headers(headers=True).generate()
        proxies = {
            'http': proxi
        }
        request = requests.get('https://yandex.ru/images/search',
                               params={"text": text,
                                       "nomisspell": 1,
                                       "noreask": 1,
                                       "isize": 'medium'
                                       },
                               headers=headers,
                               proxies=proxies,
                               cookies={'cookie': 'ob=; cpw=1300; cph=836; font_loaded=YSv1; _ym_uid=1552398047283643042; gdpr=0; mda=0; _ym_uid=1552398047283643042; my=YycCAAEA; yandex_login=konradmaklaud20; L=VlZ/cXZBB05qRApdVAAEREVCUHdQakoBABUmJiIcCQQ/OgwRJVp2.1589447579.14233.323468.b9a19c4bb7deb16f55604a246b2acfb2; yuidss=4726154681552398047; fuid01=5f588c554c4837a3.qHzlvyR7stedI6hoUCdD3JMVta9ejh0EwtU9WIqR2O0IdLfqHqFZvcNtdSFLDalhmlPWgwT35W7bW0tO2LIjCmtlLH0WE5mIhoOeueGxlh9KNw6Mi6mMW7NfbxM6kWhR; _ym_d=1601638100; yandexuid=4726154681552398047; ymex=1606161847.oyu.2263217541603569109#1900347366.yrts.1584987366#1920654614.yrtsi.1605294614; amcuid=5147372381612695498; is_gdpr=0; is_gdpr_b=CMudIBC9HCgC; sae=0:63E2B18F-17A0-40D5-81A2-CF56CEA01023:p:21.2.4.165:w:d:RU:20190312; yandex_gid=11; i=/ZuPFUpKiTZbug0wcddB0MCMG7ebIHO3Ksl3dKK3YoZ6tcs7iQIVi3RBF3S4oA+tnCQsJnCUNhNn6pJtlVpXGSXPVrk=; _ym_isad=2; Session_id=3:1621334384.5.0.1589202347615:EqDUsA:51.1|131992251.245232.2.2:245232|234725.566473.pd8NlCAr9JsNWhhx5KZaUqRL88Q; sessionid2=3:1621334384.5.0.1589202347615:EqDUsA:51.1|131992251.245232.2.2:245232|234725.566473.pd8NlCAr9JsNWhhx5KZaUqRL88Q; zm=m-white_bender-redesign.gen.webp.css-https%3As3home-static_wfEv__JqBHykiwMJ46TnpQmwMzU%3Al; ys=ead.2FECB7CF#def_bro.1#svt.1#mclid.2270456#wprid.1621359548589527-1721062715184825125300103-production-app-host-vla-web-yp-109#ybzcc.ru#newsca.native_cache; yabs-frequency=/5/7W0_0Btxes2WcffW/wGDpS9G0003iG25z_d9mb0000En0OUht6t2I0000x41Xd05pS9G0003iG474v79mb0000En0OGmESt2K0000x411l0bpS9G0003iG64__t9mb0000En0OHoWFss60000x421Yv5oS9G0003iG6403NDmb0000En0OVLmwMwD0000x41XDd51ROO0003iG2V3yN9mb0000En0OO9XGMs60000x40Zu0jpS9G0003iGE0t2dDmb0000En0e63KFMs60000x41W/; yp=1636826439.cld.2270452#1636826439.brd.6301000000#1621618794.clh.2270456#1637110941.szm.1%3A1920x1080%3A756x800#1932583857.sad.1585143279%3A1617223857%3A7#1624037631.csc.2#1904294659.multib.1#1904562224.2fa.1#1904807579.udn.cDprb25yYWRtYWtsYXVkMjA%3D#1621595534.stltp.serp_bk-map_1_1590059534#1634271249.mu.0#1621503280.ygu.1#1621427552.mcv.0#1621427552.mct.null#1621427552.mcl.17j6ajl#1621443794.nps.2817118054%3Aclose; _ym_d=1621359618; cycada=mXxsQX3dHJO3c0MtjTgsx6Cvs5LkvqGjPMTp1Z0vzOM='}
                               )

        soup = BeautifulSoup(request.text, 'html.parser')
        items_place = soup.find('div', {"class": "serp-list"})
        output = list()
        items = items_place.find_all("div", {"class": "serp-item"})
        for item in items:
            data = json.loads(item.get("data-bem"))
            image = data['serp-item']['img_href']
            try:
                r_chek = requests.get(image, timeout=7)


                if r_chek.status_code == 200 and "scontent" not in str(image) \
                        and "uploads" not in str(image):
                    output.append(Result(image))
            except Exception:
                pass
        return output


def news():

    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def normalize(text):
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    vectorizer = TfidfVectorizer(tokenizer=normalize)

    def cosine_sim(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0, 1]

    r1_lenta = requests.get('https://lenta.ru/' + datetime.datetime.now().strftime('%Y/%m/%d'))
    s1_lenta = BeautifulSoup(r1_lenta.text, 'lxml')
    all_links_lenta = ["https://lenta.ru" + i.find('a', class_='titles').get('href') for i in
                       s1_lenta.find_all('div', class_='item news b-tabloid__topic_news')]
    news_lenta_list = []
    for link_lenta in list(reversed(all_links_lenta))[:100]:
        r2_lenta = requests.get(link_lenta)
        s2_lenta = BeautifulSoup(r2_lenta.text, 'lxml')
        text_lenta_ = [i.text for i in s2_lenta.find('div', itemprop="articleBody").find_all('p')]
        title_lenta = s2_lenta.find('h1').text.strip()
        if 300 < len(" ".join(text_lenta_[:2])) < 700:
            text_lenta = " ".join(text_lenta_[:2])
        elif 300 < len(" ".join(text_lenta_[:1])) < 700:
            text_lenta = " ".join(text_lenta_[:1])
        elif 300 < len(" ".join(text_lenta_[:3])) < 700:
            text_lenta = " ".join(text_lenta_[:3])
        else:
            text_lenta = " ".join(text_lenta_)
            text_lenta = nltk.sent_tokenize(text_lenta)
            if len(text_lenta) > 5:
                text_lenta = " ".join(text_lenta[:5])
            else:
                text_lenta = " ".join(text_lenta)

        news_lenta_list.append([text_lenta, title_lenta])

    print(len(news_lenta_list))

    mname = r"C:\Users\Alexander\Downloads\mbart"

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 172,
        "train_batch_size": 2,
        "num_train_epochs": 1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 172,
        "manual_seed": 4,
        "save_steps": -1,
    }

    model_rewrite = Seq2SeqModel(
        encoder_decoder_type="mbart",
        encoder_decoder_name=mname,
        args=model_args,
        use_cuda=False,
    )
    predict_lenta_list = []
    rub_lenta = []
    c = 0
    random.shuffle(news_lenta_list)
    for for_predict_ in news_lenta_list:
        if c < 5:
            for_predict = for_predict_[0]
            predict_lenta = model_rewrite.predict([for_predict])

            predict_lenta = " ".join(predict_lenta)
            pat = r"в (понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье), [0-9]+ (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря), "
            sim = cosine_sim(for_predict, predict_lenta)
            # print('__sim: ', sim, len(for_predict), len(predict_lenta))
            # print("__20: ", predict_lenta[:20], for_predict[:20])

            MAX_SEQUENCE_LENGTH = 1075
            with open(r"C:\Users\Alexander\Downloads\tokenizer.pickle", 'rb') as handle:
                tokenizer = pickle.load(handle)
            new_model = keras.models.load_model(r"C:\Users\Alexander\Downloads\lenta_class_model.h5")
            seq = tokenizer.texts_to_sequences([predict_lenta])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
            pred = new_model.predict(padded)
            labels = sorted(
                ['Политика', 'Регионы', 'Происшествия', 'ТВ и радио', 'Музыка', 'Следствие и суд', 'Общество',
                 'Украина', 'Вирусные ролики', 'Футбол', 'Преступность', 'Госэкономика', 'Оружие', 'Криминал',
                 'Бизнес',
                 'Интернет', 'Экономика', 'Гаджеты', 'Кино', 'Конфликты', 'Люди', 'События', 'Внешний вид',
                 'Хоккей',
                 'Квартира', 'Рынки', 'Мир', 'Театр', 'Деньги', 'Летние виды', 'Белоруссия', 'Звери', 'Зимние виды',
                 'Явления', 'Город', 'Полиция и спецслужбы', 'Наука', 'Игры', 'Бокс и ММА', 'Искусство', 'Стиль',
                 'Пресса', 'Космос', 'Coцсети', 'Инструменты', 'Еда', 'Техника', 'Офис', 'История',
                 'Деловой климат',
                 'Мировой бизнес', 'Средняя Азия', 'Дача', 'Россия', 'Молдавия', 'Закавказье', 'Мнения',
                 'Достижения',
                 'Движение', 'Книги', 'Прибалтика', 'Москва', 'Жизнь', 'Часы', 'Софт', 'Мемы'])

            b = labels[np.argmax(pred)]
            predict_lenta = re.sub(pat, '', predict_lenta)
            if len(for_predict) < 1000 and len(predict_lenta) < 1000 \
                    and predict_lenta[:20] != for_predict[:20] and 0.1 < sim < 0.6:
                if b not in rub_lenta:
                    c += 1
                    predict_lenta_list.append(predict_lenta)
                    rub_lenta.append(b)

    assert len(rub_lenta) == len(predict_lenta_list)

    r_rzn = requests.get("https://62info.ru/news/ryazan/")
    s_rzn = BeautifulSoup(r_rzn.text, 'lxml')
    all_links_rzn = ["https://62info.ru" + i.get('href') for i in
                     s_rzn.find('div', class_='infinite_scroll').find_all('a', class_='td_none')[:2]]
    title_rzn_list = []
    text_rzn_list = []
    for link_rzn in all_links_rzn:
        r2_rzn = requests.get(link_rzn)
        s2_rzn = BeautifulSoup(r2_rzn.text, 'lxml')
        text_rzn_ = [i.text for i in s2_rzn.find('div', itemprop="articleBody").find_all('p')]
        if 300 < len(" ".join(text_rzn_[:2])) < 700:
            text_rzn = " ".join(text_rzn_[:2])
        elif 300 < len(" ".join(text_rzn_[:1])) < 700:
            text_rzn = " ".join(text_rzn_[:1])
        elif 300 < len(" ".join(text_rzn_[:3])) < 700:
            text_rzn = " ".join(text_rzn_[:3])
        else:
            text_rzn = " ".join(text_rzn_)
        text_rzn_list.append(text_rzn)
        title_rzn = s2_rzn.find('h1', itemprop="headline").text
        title_rzn_list.append(title_rzn)
    predict_rzn_list = []
    rub_rzn = []

    for for_predict in text_rzn_list:
        predict_rzn = model_rewrite.predict([for_predict])
        predict_rzn = " ".join(predict_rzn)
        pat = r"в (понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье), [0-9]+ (января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря), "
        predict_rzn = re.sub(pat, '', predict_rzn)

        MAX_SEQUENCE_LENGTH = 1075
        with open(r"C:\Users\Alexander\Downloads\tokenizer.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)
        new_model = keras.models.load_model(r"C:\Users\Alexander\Downloads\lenta_class_model.h5")
        seq = tokenizer.texts_to_sequences([predict_rzn])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = new_model.predict(padded)
        labels = sorted(
            ['Политика', 'Регионы', 'Происшествия', 'ТВ и радио', 'Музыка', 'Следствие и суд', 'Общество',
             'Украина', 'Вирусные ролики', 'Футбол', 'Преступность', 'Госэкономика', 'Оружие', 'Криминал',
             'Бизнес',
             'Интернет', 'Экономика', 'Гаджеты', 'Кино', 'Конфликты', 'Люди', 'События', 'Внешний вид',
             'Хоккей',
             'Квартира', 'Рынки', 'Мир', 'Театр', 'Деньги', 'Летние виды', 'Белоруссия', 'Звери', 'Зимние виды',
             'Явления', 'Город', 'Полиция и спецслужбы', 'Наука', 'Игры', 'Бокс и ММА', 'Искусство', 'Стиль',
             'Пресса', 'Космос', 'Coцсети', 'Инструменты', 'Еда', 'Техника', 'Офис', 'История',
             'Деловой климат',
             'Мировой бизнес', 'Средняя Азия', 'Дача', 'Россия', 'Молдавия', 'Закавказье', 'Мнения',
             'Достижения',
             'Движение', 'Книги', 'Прибалтика', 'Москва', 'Жизнь', 'Часы', 'Софт', 'Мемы'])

        b = labels[np.argmax(pred)]
        rub_rzn.append(b)

        predict_rzn_list.append(predict_rzn)
    try:
        parser = YandexImage()
        res_rzn = [random.choice([i.url for i in parser.search(title_rzn_list[0])]),
                   random.choice([i.url for i in parser.search(title_rzn_list[1])])]
    except:
        parser = YandexImage1()
        res_rzn = [random.choice([i.url for i in parser.search(title_rzn_list[0])]),
                   random.choice([i.url for i in parser.search(title_rzn_list[1])])]


    style = "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]),
                                                        random.choice([i for i in range(30, 40)]),
                                                        random.choice([i for i in range(30, 40)]),
                                                        random.choice([i for i in range(69, 80)]),
                                                        random.choice([i for i in range(60, 70)]),
                                                        random.choice([i for i in range(40, 50)]),
                                                        random.choice([i for i in range(60, 70)]),
                                                        random.choice([i for i in range(40, 50)]))

    def title_news(lenta_list, rzn_list):
        predict_lenta_list = lenta_list + rzn_list

        tokenizer = MT5Tokenizer.from_pretrained(r"C:\Users\Alexander\Downloads\mt5-small")
        model = MT5ForConditionalGeneration.from_pretrained(r"C:\Users\Alexander\Downloads\mt5-small")

        predictions_title_lenta_list = []
        for for_title_predict in predict_lenta_list:
            input_ids = tokenizer.encode(for_title_predict, return_tensors='pt')

            beam_output = model.generate(
                input_ids,
                max_length=50,
                num_beams=5,
                early_stopping=True)

            predictions_title_lenta = tokenizer.decode(beam_output[0], skip_special_tokens=True)

            predictions_title_lenta_list.append(predictions_title_lenta)
        return predictions_title_lenta_list

    title_lenta_list_ = title_news(predict_lenta_list, predict_rzn_list)

    random_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(7)]

    try:
        parser = YandexImage()
        res_lenta = [random.choice([i.url for i in parser.search(title_lenta_list_[0])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[1])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[2])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[3])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[4])])]
    except Exception:
        parser = YandexImage1()
        res_lenta = [random.choice([i.url for i in parser.search(title_lenta_list_[0])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[1])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[2])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[3])]),
                     random.choice([i.url for i in parser.search(title_lenta_list_[4])])]




    template_news = """
    <img src="https://i.ibb.co/pykgvzt/logo.png" alt="" style="">
    <p style='margin: 10px auto; font-family: "Times New Roman", Times, serif; font-style: oblique;'>Ежечасное издание, <br>созданное искусственным интеллектом</p>
    <p style="font-family: Impact, Charcoal, sans-serif">Выпуск от {}</p>
     <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
     <h3 class='mt-2' style="
    font-family: 'Courier New', Courier, monospace;
    font-weight: 900;
      text-transform: uppercase;
      background: linear-gradient(90deg, rgba(117,85,162,1) 0%, rgba(244,53,53,1) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: #0B2349;
      display: table;
      margin: 20px auto;" align="center">Новости России и мира</h3></div>
        <div class="block_pogoda">
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>
            </div>
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
        </div>
        
        <div class="block_pogoda">
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
        </div>
        
        
        <div class="block_pogoda">
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
        </div>
        
        <div class="block_pogoda">
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
        </div>
        
        
        <div class="block_pogoda">
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
        </div>
                <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
                 <h3 class='mt-2' style="
        font-family: 'Courier New', Courier, monospace;
        font-weight: 900;
          text-transform: uppercase;
          background: linear-gradient(90deg, rgba(71,187,83,1) 0%, rgba(53,178,209,1) 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          color: #0B2349;
          display: table;
          margin: 20px auto;" align="center">Региональные новости — Рязань</h3>
        </div>
        
        <div class="block_pogoda">
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
        </div>
        
        
        <div class="block_pogoda">
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: {}">
                <p style='color:{}; font-size: 15px; margin-top: 15px;'>#{}</p>

            </div>
            <div class="block__text_pogoda">
                <h2>{} </h2>
                <p style="text-align:left;">{} </p>
            </div>
        </div>

    """.format(datetime.datetime.now().strftime('%d.%m.%Y %H:00'),
               res_lenta[0], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[0], rub_lenta[0], title_lenta_list_[0], predict_lenta_list[0],
               title_lenta_list_[1], predict_lenta_list[1], res_lenta[1], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[1], rub_lenta[1],
               res_lenta[2], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[2], rub_lenta[2], title_lenta_list_[2], predict_lenta_list[2],
               title_lenta_list_[3], predict_lenta_list[3], res_lenta[3], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[3], rub_lenta[3],
               res_lenta[4], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[4], rub_lenta[4], title_lenta_list_[4], predict_lenta_list[4],
               title_lenta_list_[5], predict_rzn_list[0], res_rzn[0], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[5], rub_rzn[0],
               res_rzn[1], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), random_color[6], rub_rzn[1], title_lenta_list_[6], predict_rzn_list[1])

    return template_news


def persons():
    text = ''
    r = requests.get('https://lenta.ru/')
    s = BeautifulSoup(r.text, 'lxml')

    all_links_lenta = ["https://lenta.ru" + i.find('a').get('href') for i in
                       s.find('div', class_='b-yellow-box__wrap').find_all('div', class_='item')]
    news_list = []
    for link_lenta in all_links_lenta:
        try:
            r2 = requests.get(link_lenta)
            s2 = BeautifulSoup(r2.text, 'lxml')
            text_lenta = " ".join([i.text for i in s2.find('div', itemprop="articleBody").find_all('p')][:2])
            text += text_lenta
        except Exception:
            pass

    r3 = requests.get('https://www.gazeta.ru/')
    s3 = BeautifulSoup(r3.text, 'lxml')

    all_links_gazeta = ["https://www.gazeta.ru" + i.find('a').get('href') for i in
                        s3.find('section', class_='tile_cent b-white sausage').find_all('li',
                                                                                        class_='sausage-list-item')]
    for link_gazeta in all_links_gazeta:
        try:
            r4 = requests.get(link_gazeta)
            s4 = BeautifulSoup(r4.text, 'lxml')
            text += s4.find('div', itemprop="articleBody").find('p').text
        except AttributeError:
            pass

    r5 = requests.get('https://ria.ru/')
    s5 = BeautifulSoup(r5.text, 'lxml')

    all_links_ria = [i.find('a').get('href') for i in
                     s5.find('div', class_='floor__cell', attrs={"data-block-position": "3"}).
                         find_all('div', class_='cell-list__item m-no-image', attrs={"data-article-type": "article"})]

    for link_ria in all_links_ria:
        try:
            r6 = requests.get(link_ria)
            s6 = BeautifulSoup(r6.text, 'lxml')
            pre_text = s6.find('div', class_='article__block', attrs={"data-type": "text"}).text
            ind1 = pre_text.find('— РИА Новости.')
            pre_text = pre_text[ind1 + len('— РИА Новости. '):]
            text += pre_text
        except AttributeError:
            pass

    r7 = requests.get('https://www.rbc.ru/')
    s7 = BeautifulSoup(r7.text, 'lxml')

    all_links_rbc = [i.find('a').get('href') for i in s7.find('div', class_='main__list').
        find_all('div', class_='main__feed js-main-reload-item') if 'https://www.rbc.ru' in i.find('a').
                         get('href')]

    for link_rbc in all_links_rbc:
        try:
            r8 = requests.get(link_rbc)
            s8 = BeautifulSoup(r8.text, 'lxml')
            text += s8.find('div', itemprop="articleBody").find('p').text
        except AttributeError:
            pass

    r_meduza = requests.get('https://meduza.io/api/w5/search?chrono=news&page=0&per_page=24&locale=ru')
    js_meduza = dict(json.loads(r_meduza.text))
    all_links_meduza = ["https://meduza.io/" + i for i in js_meduza['collection']][:5]

    for link_meduza in all_links_meduza:
        try:
            r2_meduza = requests.get(link_meduza)
            s2_meduza = BeautifulSoup(r2_meduza.text, 'lxml')
            text_meduza = [i.text.replace('\xa0', ' ').replace("\u202f", ' ') for i in s2_meduza.find_all('p')][1:4]
            text_meduza = " ".join(text_meduza)
            text += text_meduza
        except AttributeError:
            pass

    r_tass = requests.get("https://tass.ru/")
    s_tass = BeautifulSoup(r_tass.text, 'lxml')

    all_links_tass = ["https://tass.ru/" + i.find('a').get('href') for i in
                      s_tass.find('ul', class_='popular-news__list').find_all('li', class_='popular-news__item')]

    for link_tass in all_links_tass:
        try:
            r2_tass = requests.get(link_tass)
            s2_tass = BeautifulSoup(r2_tass.text, 'lxml')

            text_tass = [i.text.replace('\xa0', ' ') for i in s2_tass.find('div', class_='text-block').find_all('p')][
                        :3]
            text_tass = " ".join(text_tass)
            ind1 = text_tass.find('/ТАСС/')
            if ind1 != -1:
                text_tass = text_tass[ind1 + len('/ТАСС/. '):]
            text += text_tass
        except AttributeError:
            pass

    nlp = spacy.load("ru_core_news_sm")
    l1 = []
    l2 = []

    doc = nlp(text)
    for ent in doc.ents:

        if ent.label_ == 'PER':
            l1.append(ent.text)

    df = pd.DataFrame({'name': l1})

    def clean_text(text_):
        text_ = text_.split()
        if len(text_) > 1:
            words = []
            for i in text_:
                try:
                    r_f = requests.get("https://how-to-all.com/морфология:{}".format(i))
                    s_f = BeautifulSoup(r_f.text, 'lxml')
                    word = s_f.find_all('div', class_='chr')[-1].find('div', class_='chrd').text.capitalize()
                    words.append(word)
                except:
                    pass
            return " ".join(words)

    df_sort = df['name'].value_counts(sort=True)
    df_sort = pd.DataFrame(df_sort)
    df_sort['PER'] = df_sort.index
    df_sort = df_sort[:10]
    df_sort['PER'] = df_sort.apply(lambda x: clean_text(x['PER']), axis=1)
    df_sort = df_sort.loc[df_sort['PER'] != 'Владимир Путин'].reset_index(drop=True)
    df_sort = df_sort.loc[df_sort['PER'] != 'Путин'].reset_index(drop=True)
    df_sort = df_sort.loc[df_sort['PER'] != 'Владимира Путина'].reset_index(drop=True)
    df_sort = df_sort.drop_duplicates(subset="PER").reset_index(drop=True)
    df_sort = df_sort.dropna().reset_index(drop=True)
    try:
        parser = YandexImage()

        res = [random.choice([i.url for i in parser.search(df_sort['PER'][0])]),
               random.choice([i.url for i in parser.search(df_sort['PER'][1])]),
               random.choice([i.url for i in parser.search(df_sort['PER'][2])])]
    except Exception:
        parser = YandexImage1()

        res = [random.choice([i.url for i in parser.search(df_sort['PER'][0])]),
               random.choice([i.url for i in parser.search(df_sort['PER'][1])]),
               random.choice([i.url for i in parser.search(df_sort['PER'][2])])]


    # style = "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]))

    template_persons = """
       <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
     <h3 class='mt-2' style="
    font-family: 'Courier New', Courier, monospace;
    font-weight: 900;
      text-transform: uppercase;
      background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: #0B2349;
      display: table;
      margin: 20px auto;"
     align="center">Главные персоны к этому часу — {}</h3>
    </div>
            <div class="row row-cols-1 row-cols-md-3">
        <div class="col">
        <div class="card mb-3" style='border:none'>
        <p style="text-align:center;">
             <div class="img_wrap">
            <img src="{}" alt="normal" style="border-radius:{}"/>
          </div>
        </p>
        <p style="text-align:center;">{}</p>
        </div>
        </div>
        <div class="col">
        <div class="card mb-3" style='border:none'>
        <p style="text-align:center;">
          <div class="img_wrap">
            <img src="{}" alt="normal" style="border-radius:{}"/>
          </div>
        </p>
        <p style="text-align:center;">{}</p>
        </div>
        </div>
        <div class="col">
        <div class="card mb-3" style='border:none'>
        <p style="text-align:center;">
          <div class="img_wrap">
            <img src="{}" alt="normal" style="border-radius:{}"/>
          </div>
        </p>
        <p style="text-align:center;">{}</p>
        </div>
        </div>
         </div>

    """.format(datetime.datetime.now().strftime("%H:%M"),
               res[0], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), df_sort['PER'][0],
               res[1], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), df_sort['PER'][1],
               res[2], "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), df_sort['PER'][2])
    return template_persons


def pogoda():
    r = requests.get('https://pogoda7.ru/prognoz/gorod764-Russia-Ryazanskaya_oblast-Ryazan')
    s = BeautifulSoup(r.text, 'lxml')
    all_info = s.find_all('div', class_='table-row-day dayline')[1]
    try:
        date_day = all_info.find('div', class_='dayweek_weekend').text
    except:
        date_day = all_info.find('div', class_='dayweek_date').text
    days_dict = {'Пн': "понедельник", 'Вт': "вторник", 'Ср': "среда", 'Чт': "четверг",
                 'Пт': "пятница", 'Сб': "суббота", 'Вс': "воскресенье"}

    if date_day[0] == '0':
        date_day = date_day[1:]
    date_day_week = all_info.find('div', class_='dayweek_week').text

    for i in days_dict.keys():
        if i in date_day_week:
            date_day_week = days_dict[i]

    night_info = all_info.find_all('div', class_='table-row-time')[0]
    morning_info = all_info.find_all('div', class_='table-row-time')[1]
    day_info = all_info.find_all('div', class_='table-row-time')[2]
    evening_info = all_info.find_all('div', class_='table-row-time')[3]

    night_get_info = night_info.find('div', class_='table-cell temperature_cell').get('title')
    night_get_list = [i.replace('\r', '').split(':') for i in night_get_info.split('\n')]
    night_dict = {}
    for info in night_get_list:

        if len(info) > 1:
            night_dict[info[0]] = info[1].strip()

    morning_get_info = morning_info.find('div', class_='table-cell temperature_cell').get('title')
    morning_get_list = [i.replace('\r', '').split(':') for i in morning_get_info.split('\n')]
    morning_dict = {}
    for info in morning_get_list:

        if len(info) > 1:
            morning_dict[info[0]] = info[1].strip()

    day_get_info = day_info.find('div', class_='table-cell temperature_cell').get('title')
    day_get_list = [i.replace('\r', '').split(':') for i in day_get_info.split('\n')]
    day_dict = {}
    for info in day_get_list:

        if len(info) > 1:
            day_dict[info[0]] = info[1].strip()

    evening_get_info = evening_info.find('div', class_='table-cell temperature_cell').get('title')
    evening_get_list = [i.replace('\r', '').split(':') for i in evening_get_info.split('\n')]
    evening_dict = {}
    for info in evening_get_list:

        if len(info) > 1:
            evening_dict[info[0]] = info[1].strip()

    f = requests.get('https://ws3.morpher.ru/russian/declension?s={}&format=json'.format(date_day_week))
    data_morph = json.loads(f.text)

    assert len(night_dict) == len(morning_dict) == len(day_dict) == len(evening_dict)

    night_temp = ', '.join(night_dict['Температура'].split(' .. '))
    morning_temp = ', '.join(morning_dict['Температура'].split(' .. '))
    day_temp = ', '.join(day_dict['Температура'].split(' .. '))
    evening_temp = ', '.join(evening_dict['Температура'].split(' .. '))

    night_osad = night_dict['Осадки']
    morning_osad = morning_dict['Осадки']
    day_osad = day_dict['Осадки']
    evening_osad = evening_dict['Осадки']

    night_obl = night_dict['Облачность']
    morning_obl = morning_dict['Облачность']
    day_obl = day_dict['Облачность']
    evening_obl = evening_dict['Облачность']

    night_veter = night_dict['Ветер']
    morning_veter = morning_dict['Ветер']
    day_veter = day_dict['Ветер']
    evening_veter = evening_dict['Ветер']

    what_wait = []
    for i in [day_osad, evening_osad, morning_osad]:
        if 'дождь' in i:
            what_wait.append(i.replace('возможен ', ''))
    if len(what_wait) == 0:

        for i in [day_obl, evening_obl, morning_obl]:

            if 'пасмурно' in i:
                what_wait.append('пасмурная погода, без осадков')

        for i in [day_obl, evening_obl, morning_obl]:
            if 'ясно' in i:
                what_wait.append('ясная погода, без осадков')
    what_wait_veter = ''

    if int(day_veter.split(', ')[-1].split()[0]) > 8:
        what_wait_veter += ' и ' + (day_veter.split(', ')[0])

    up_or_down1 = [float(i) for i in day_temp.replace('+', '').replace(',', '').replace(' °C', '').split()]
    up_or_down1_mean = (up_or_down1[0] + up_or_down1[1]) / 2

    up_or_down2 = [float(i) for i in evening_temp.replace('+', '').replace(',', '').replace(' °C', '').split()]
    up_or_down2_mean = (up_or_down2[0] + up_or_down2[1]) / 2

    up_or_down_res = ['опустится' if up_or_down2_mean < up_or_down1_mean else 'поднимится']
    max_veter = max(int(day_veter.split(', ')[2].replace(' м/с', '')),
                    int(morning_veter.split(', ')[2].replace(' м/с', '')),
                    int(night_veter.split(', ')[2].replace(' м/с', '')),
                    int(evening_veter.split(', ')[2].replace(' м/с', '')), )

    if max_veter == int(day_veter.split(', ')[2].split()[0]):
        max_veter = ''
    else:
        max_veter = "Местами порывы ветра могут достигать {} метров в секунду. ".format(max_veter)
    v_or_vo = ''
    if data_morph['В'] == 'вторник':
        v_or_vo = 'о'
    try:
        what_wait_res = random.choice(what_wait)
    except:
        what_wait_res = 'облачная погода'
    title_pogoda = "Завтра в Рязани ожидается {}{}".format(what_wait_res, what_wait_veter,)
    template = """В{} {}, {}, в Рязани ожидается {}{}.<br>
    По данным синоптиков, температура воздуха ночью составит {}. Утром {}. <br>Днём от {} до {} градусов. К вечеру столбик термометра {} до {}.<br>
    Ветер {}, {}. {}Давление {} Относительная влажность воздуха {}.""".format(v_or_vo, data_morph['В'], date_day,
                                                                              what_wait_res, what_wait_veter,
                                                                              night_temp,
                                                                              morning_temp, day_temp.split(', ')[0],
                                                                              day_temp.split(', ')[1].replace(' °C',
                                                                                                              ''),
                                                                              *up_or_down_res, evening_temp, ', '.join(
            day_veter.split(', ')[1].split(',')),
                                                                              day_veter.split(', ')[2], max_veter,
                                                                              day_dict['Давление'],
                                                                              day_dict['Влажность'])
    try:
        parser = YandexImage()
        photo = random.choice([i.url for i in parser.search(what_wait_res)])
    except Exception:
        parser = YandexImage1()
        photo = random.choice([i.url for i in parser.search(what_wait_res)])

    template_pogoda = """
      <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
     
     
     <h3 class='mt-2' style="
    font-family: 'Courier New', Courier, monospace;
    font-weight: 900;
      text-transform: uppercase;
      background: linear-gradient(90deg, rgba(29,242,253,1) 0%, rgba(165,69,252,1) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: #0B2349;
      display: table;
      margin: 20px auto;"
     align="center">Погода на завтра в Рязани</h3>
      </div>
    <div class="block_pogoda">
    <div class="img_wrap">
        <img src="{}" alt="" style="border-radius:{}">
    </div>
    <div class="block__text_pogoda">
    <h2>{}</h2>
        <p style="text-align:left;">{}</p>
            </div>
        </div>
        
     <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
     <h3 class='mt-2' style="
    font-family: 'Courier New', Courier, monospace;
    font-weight: 900;
      text-transform: uppercase;
      background: linear-gradient(90deg, rgba(190,106,232,1) 0%, rgba(69,106,215,1) 34%, rgba(207,137,187,1) 66%, rgba(25,228,250,1) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: #0B2349;
      display: table;
      margin: 20px auto;"
     align="center">Ситуация с коронавирусом в Рязанской области</h3>
    </div>
    <div id='h5' style='text-align: left'></div>
        
    """.format(photo, "{}% {}% {}% {}% / {}% {}% {}% {}%;".format(random.choice([i for i in range(69, 80)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(30, 40)]), random.choice([i for i in range(69, 80)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)]), random.choice([i for i in range(60, 70)]), random.choice([i for i in range(40, 50)])), title_pogoda, template)

    return template_pogoda


def recept():
    trump = pipeline('text-generation', model=r"D:\С\recept",
                     tokenizer='sberbank-ai/rugpt3small_based_on_gpt2', config={'max_length': 1000, 'temperature': .5})

    initial_seed = "РЕЦЕПТ:"

    recepts = trump(initial_seed, max_length=350, repetition_penalty=1.2, no_repeat_ngram_size=3,
                    early_stopping=True, min_length=300, do_sample=True, temperature=1.2)[0]['generated_text']

    recepts = recepts.replace('\xa0', ' ').replace('\n', '\n')
    lst_rec = []
    for recept in recepts.split('°'):
        if len(recept) > 10:
            ind1 = recept.find('²')
            title = recept[:ind1].replace('РЕЦЕПТ:', '').strip().capitalize()

            ind2 = recept.find('ИНГРЕДИЕНТЫ')
            recept_text = recept[ind2:].strip()
            if "ПРИЯТНОГО АППЕТИТА" not in recept_text:
                tchk = recept_text.rfind('.')

                recept_text = recept_text[:tchk] + '.'
            if "ПРИЯТНОГО АППЕТИТА" not in recept_text:
                recept_text += '\n' + "ПРИЯТНОГО АППЕТИТА!"

            try:
                parser = YandexImage()
                photo = random.choice([i.url for i in parser.search(title)])
            except Exception:
                parser = YandexImage1()
                photo = random.choice([i.url for i in parser.search(title)])

            recept_text = recept_text.replace('\n', "<br>")
            ind1 = recept_text.find("ИНГРЕДИЕНТЫ") + len("ИНГРЕДИЕНТЫ")
            ind2 = recept_text.find("ИНСТРУКЦИЯ ПРИГОТОВЛЕНИЯ")
            text2 = recept_text[ind1:ind2]

            text2 = text2.replace('<br>', ' ').split(';')
            recept_text = "ИНГРЕДИЕНТЫ<br>" + "&#x1F6A9;" + "<br>&#x1F6A9;".join(text2)[
                                                            :-len("&#x1F6A9;") - 1] + recept_text[ind2:]
            recept_text = recept_text.replace("ИНГРЕДИЕНТЫ", "<font color='#10d740'>ИНГРЕДИЕНТЫ</font>").replace(
                "ИНСТРУКЦИЯ ПРИГОТОВЛЕНИЯ", "<br><font color='#19e4fa'>ИНСТРУКЦИЯ ПРИГОТОВЛЕНИЯ</font>").replace(
                "ПРИЯТНОГО АППЕТИТА!", "<font color='#ff76d8'>ПРИЯТНОГО АППЕТИТА!</font>")
            if recept_text.count('&#x1F6A9;') > 15:
                recept()
            template_recept = """
            <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">


                 <h3 class='mt-2' style="
                font-family: 'Courier New', Courier, monospace;
                font-weight: 900;
                  text-transform: uppercase;
                  background: linear-gradient(90deg, rgba(29,253,68,1) 0%, rgba(130,50,224,1) 49%, rgba(227,194,32,1) 100%);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;
                  color: #0B2349;
                  display: table;
                  margin: 20px auto;" align="center">Попробуй приготовить — нейросетевые рецепты</h3>
                  </div>

                    <div class="block_pogoda">
                    <div class="block__text_pogoda">
                        <h2>&#x1F37D;&#xFE0F;{}</h2>
                        <p style="text-align:left;">{}</p>
                    </div>
                    <div class="img_wrap">
                        <img src="{}" alt="" style="border-radius: 69% 31% 33% 79% / 64% 44% 67% 44%;">
                    </div>
                </div>

            """.format(title, recept_text, photo)
            return template_recept


def peiz():
    bad_words_df = pd.read_csv(r"C:\Users\Alexander\Downloads\bad_words.csv")
    res_per = [i for i in bad_words_df['word']]
    tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    bad_words_list = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in res_per]
    model = GPT2LMHeadModel.from_pretrained(r"D:\С\peiz")
    initial_seed = ["Крупные капли дождя", "Белое солнце", "Осенная тишина ночи была",
                    "Запах леса был", "Лесная поляна цветов", "Широкое поле",
                    "В глухой чаще", "Солнечный блик радостно", "Зелёная трава радостно",
                    "Тонкие ветви акации, словно", "Вековой дуб непоколибимо",
                    "Грозовая туча, похожая", "Холодная пелена неба, как",
                    "Чёрная полоса горизонта наводила", "Жёлтые листья казались",
                    "Серая дымка спускалась с", "Ночной туман окутал"]
    initial_seed = random.choice(initial_seed)
    input_ids = tokenizer(initial_seed, return_tensors="pt").input_ids
    bad_words_list.append([28568, 1123])
    bad_words_list.append([28568])
    bad_words_list.append([11699, 383])
    bad_words_list.append([618, 3646])
    bad_words_list.append([8215, 225])
    bad_words_list.append([1098, 320, 2810, 266])
    bad_words_list.append([672, 1408, 3372])
    bad_words_list.append([19115, 465, 669])
    bad_words_list.append([3578, 27976])
    bad_words_list.append([3578, 3685])
    bad_words_list.append([265, 7454, 1947])
    bad_words_list.append([7223, 303, 1254])
    bad_words_list.append([7223, 303])
    bad_words_list.append([7223, 2292])
    bad_words_list.append([267, 324, 989])
    bad_words_list.append([267, 324, 989, 291])
    bad_words_list.append([267, 324])
    bad_words_list.append([348, 15796])
    bad_words_list.append([3399, 5055, 451])
    bad_words_list.append([47101])
    bad_words_list.append([18758, 293])
    bad_words_list.append([1786, 451])
    bad_words_list.append([267, 324])

    sample_output = model.generate(
        input_ids,
        max_length=200,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        early_stopping=True,
        bad_words_ids=bad_words_list
    )

    this = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    this = " ".join(nltk.sent_tokenize(this)[:-1])
    this = this.replace('\xa0', ' ').replace('\n', '\n').replace('&nbsp;', ' ').replace('\t', ' ').replace('\\t', ' ')
    ind_1 = this.find(initial_seed)
    ind_2 = this.find("»")
    if this[ind_1 + 1] == this[ind_2]:
        this = "«" + this
    this = re.sub(r'\[\w+]', '', this)
    this = re.sub(r"\s+", " ", this)
    try:
        parser = YandexImage()
        photo = random.choice([i.url for i in parser.search(initial_seed)])
    except Exception:
        parser = YandexImage1()
        photo = random.choice([i.url for i in parser.search(initial_seed)])


    template_peiz = """
         <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
     <h3 class='mt-2' style="
    font-family: 'Courier New', Courier, monospace;
    font-weight: 900;
      text-transform: uppercase;
      background: linear-gradient(90deg, rgba(253,79,29,1) 0%, rgba(227,194,32,1) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      color: #0B2349;
      display: table;
      margin: 20px auto;"
     align="center">Литературная страница</h3></div>
    
    <h2 style='font-family: ‘Lucida Console’, Monaco, monospace; font-style: italic;'>Зарисовки о природе и жизни</h2>
    <div class='circle1' style='background: url("{}"); background-size: cover;'></div>
    <p style="text-align:left;">{}</p>

    """.format(photo, this)

    return template_peiz


def stihi():
    trump = pipeline('text-generation', model=r"D:\С\stihi",
                     tokenizer='sberbank-ai/rugpt3small_based_on_gpt2', config={'max_length': 1000, 'temperature': .5})

    initial_seed = "СТИХ: "

    stih = trump(initial_seed, max_length=350, repetition_penalty=1.2, no_repeat_ngram_size=3,
                 early_stopping=True, min_length=300, do_sample=True, temperature=1.2)[0]['generated_text']

    stih_list = stih.split('СТИХ')
    max_len = max([len(i.replace(':', '').strip()) for i in stih_list])
    for stih_ in stih_list:
        stih_ = stih_.replace(':', '').strip()
        if len(stih_) == max_len:
            ind1 = stih_.find('²')
            ind2 = stih_.find('°')
            title = stih_[:ind1].lower().replace('стих', '').replace(':', '').strip().capitalize()

            if "°" in title:
                title = '***'
            ind3 = stih_.find(title)

            if ind3 != -1:
                stih_ = stih_[ind3 + len(title):ind2].replace('²', '')
            else:
                stih_ = stih_[:ind2].replace('²', '')

            stih_ = list(stih_.replace('СТИХ: ', ''))
            for e, i in enumerate(stih_):
                # print(e, i)
                if i.isupper() is True and stih_[e - 1].isupper() is False:
                    # print(stih[e-1])
                    stih_[e - 1] = stih_[e - 1] + '\n'

            stih_ = ''.join(stih_).replace('  ', ' ').replace('Припев:', '')

            if stih_.split('\n')[-1] == '':
                stih_ = '\n'.join(stih_.split('\n')[:-2]).strip()
            elif stih_.split('\n')[-1] != '':
                stih_ = '\n'.join(stih_.split('\n')[:-1]).strip()
            try:
                if stih_[-1] not in '!.?':
                    stih_ = stih_ + ' '
                    stih_ = stih_[:-1] + '.'
                    assert len(stih_) > 200
            except Exception:
                stihi()

            #print("title: ", title)
            #print("stih: ", stih_)
            try:
                parser = YandexImage()
                photo = random.choice([i.url for i in parser.search(title)])
            except Exception:
                parser = YandexImage1()
                photo = random.choice([i.url for i in parser.search(title)])

            size = int(stih_.replace('\n', '<br>').count('<br>')) * 37
            template_stih = """
                    <div class="block_pogoda" style='margin-top: 30px'>
            <div class="block__text_pogoda">
                <h2 style='font-family: ‘Lucida Console’, Monaco, monospace; font-style: italic;'>{}</h2>
                <p style="text-align:center;">{}</p>
                </div>
                <div class="img_wrap2" style="height: {}px;">
                <img src="{}" alt="" style="border-radius: 69% 31% 33% 79% / 64% 44% 67% 44%;">
            </div>
            </div>
             <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
        
                 <h3 class='mt-2' style="
                font-family: 'Courier New', Courier, monospace;
                font-weight: 900;
                  text-transform: uppercase;
                  background: linear-gradient(90deg, rgba(190,106,232,1) 0%, rgba(69,106,215,1) 34%, rgba(207,137,187,1) 66%, rgba(25,228,250,1) 100%);
                  -webkit-background-clip: text;
                  -webkit-text-fill-color: transparent;
                  color: #0B2349;
                  display: table;
                  margin: 20px auto;"
                 align="center">Информация о редакции</h3>
                </div>
            
            
            <div class="row row-cols-1 row-cols-md-2" style='margin: 30px auto'>
              <div class="col">
                <div class="card h-100" style='border: 1px solid #9999FF; border-radius: 30px;'>
                    <div class="gradient1">
                  <img src="https://i10.fotocdn.net/s112/c6532fb5ad8eea3f/public_pin_l/2507616594.jpg" class="card-img-top" style='border-radius: 30px;'>
                  </div>
                  <div class="card-body">
                    <h5 class="card-title">MBart</h5>
                    <p class="card-text">Энкодер-декодер модель, обученная на 100 тыс. пар новостных рерайтов</p>
                  </div>
                  <div class="card-footer" style="margin-bottom: 30px; background: none; border:none;">
                    <small class="text-muted">Отдел новостей</small>
                  </div>
                </div>
              </div>
            
              <div class="col">
                <div class="card h-100" style='border: 1px solid #9999FF; border-radius: 30px;'>
                    <div class="gradient3">
                    <img src="https://i10.fotocdn.net/s112/c6532fb5ad8eea3f/public_pin_l/2507616594.jpg" class="card-img-top" style='border-radius: 30px;'>
                    </div>
                  <div class="card-body">
                    <h5 class="card-title">ruGPT-3</h5>
                    <p class="card-text">Генеративная трансформер-модель, обученная на корпусе русской литературы</p>
                  </div>
                  <div class="card-footer" style="margin-bottom: 30px; background: none; border:none;">
                    <small class="text-muted">Литературный и кулинарный отдел</small>
                  </div>
                </div>
              </div>
            </div>
                  
            
                  
             <div class="row row-cols-1 row-cols-md-3" style='margin: 30px auto'>
                 
                   <div class="col">
                <div class="card h-100" style='border: 1px solid #9999FF; border-radius: 30px;'>
                    <div class="gradient2">
                  <img src="https://i10.fotocdn.net/s112/c6532fb5ad8eea3f/public_pin_l/2507616594.jpg" class="card-img-top" style='border-radius: 30px;'>
                  </div>
                  <div class="card-body">
                    <h5 class="card-title">MT5</h5>
                    
                    <p class="card-text">Text2text модель, обученная на 500 тыс. новостей</p>
                  </div>
                  <div class="card-footer" style="margin-bottom: 30px; background: none; border:none;">
                    <small class="text-muted">Создание заголовков</small>
                  </div>
                </div>
              </div>
                 
              <div class="col">
                <div class="card h-100" style='border: 1px solid #9999FF; border-radius: 30px;'>
                    <div class="gradient4">
                  <img src="https://i10.fotocdn.net/s112/c6532fb5ad8eea3f/public_pin_l/2507616594.jpg" class="card-img-top" style='border-radius: 30px;'>
                  </div>
                  <div class="card-body">
                    <h5 class="card-title">Spacy</h5>
                    <p class="card-text">Python-библиотека для извелечения из текста имён</p>
                  </div>
                  <div class="card-footer" style="margin-bottom: 30px; background: none; border:none;">
                    <small class="text-muted">Главные персоны к текущему часу</small>
                  </div>
                </div>
              </div>
              <div class="col">
                <div class="card h-100" style='border: 1px solid #9999FF; border-radius: 30px;'>
                    <div class="gradient5">
                  <img src="https://i10.fotocdn.net/s112/c6532fb5ad8eea3f/public_pin_l/2507616594.jpg" class="card-img-top" style='border-radius: 30px;'>
                  </div>
                  <div class="card-body">
                    <h5 class="card-title">Парсеры</h5>
                    
                    <p class="card-text">Python-скрипты для сбора информации с веб-страниц</p>
                  </div>
                  <div class="card-footer" style="margin-bottom: 30px; background: none; border:none;">
                    <small class="text-muted">Отдел оперативной информации</small>
                  </div>
                </div>
              </div>
            
            </div> 
                        
            
            """.format(title, stih_.replace('\n', '<br>'), size, photo)

            return template_stih


def horoscope():
    horoscope_model = pipeline('text-generation', model=r"D:\С\goroscope",
                               tokenizer='sberbank-ai/rugpt3small_based_on_gpt2')
    start_list = ["Скоро вы узнаете, что", "Звёзды говорят, что", "Этот день сулит вам", "Не забывайте о",
                  "Самое главное сейчас - это", "Вас ожидает", "Астрологи уверены, что сегодня", "Удача улыбается вам,",
                  "Сразу с утра вас ждёт", "Утром лучше", "Возьмите себя в руки", "На любовном фронте произойдут",
                  "Нет ничего лучше, чем", "Отличное время, чтобы", "Звёзды рекомендуют",
                  "Расположение планет в этот день говорит о", "В течение всего дня вас будет преследовать"]
    initial_seed = random.choice(start_list)

    speech = horoscope_model(initial_seed, max_length=130, min_length=100, do_sample=True, temperature=1.2)[0][
        'generated_text']
    speech_list = nltk.sent_tokenize(speech)
    speech_list = speech_list[1:len(speech_list) - 1]
    full = " ".join(speech_list)
    try:
        parser = YandexImage()
        img1 = random.choice([i.url for i in parser.search(" ".join(full.split()[:5]))])
        img2 = random.choice([i.url for i in parser.search(" ".join(full.split()[:5]))])
    except Exception:
        parser = YandexImage1()
        img1 = random.choice([i.url for i in parser.search(" ".join(full.split()[:5]))])
        img2 = random.choice([i.url for i in parser.search(" ".join(full.split()[:5]))])

    template = """
    <div style='border-top: 2px solid #9999FF;'></div>
                <div style="border-top: 1px solid #9999FF;border-left: 2px solid #9999FF;border-right: 2px solid #9999FF; display: inline-block; position: relative; padding: 10px; border-radius: 40px 40px 0 0">
                 <h3 class='mt-2' style="
        font-family: 'Courier New', Courier, monospace;
        font-weight: 900;
          text-transform: uppercase;
          background: linear-gradient(90deg, rgba(231,232,106,1) 0%, rgba(145,25,250,1) 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          color: #0B2349;
          display: table;
          margin: 20px auto;" align="center">Гороскопы</h3>
        </div>
                <div class="block_pogoda">
                     <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: 75% 37% 39% 76% / 65% 45% 60% 49%;">
                <p style='color:#F2F3DC; font-size: 15px; margin-top: 15px;'>#Политика</p>

            </div>
            <div class="block__text_pogoda">
                <h2>Ваш гороскоп на сегодня</h2>
                <p style="text-align:left;">{}</p>
            </div>
            <div class="img_wrap">
                <img src="{}" alt="" style="border-radius: 75% 37% 39% 76% / 65% 45% 60% 49%;">
                <p style='color:#F2F3DC; font-size: 15px; margin-top: 15px;'>#Политика</p>
            </div>
        </div>
    
    """.format(img1, full, img2)
    return template


four = recept()
five = peiz()
six = stihi()
seven = horoscope()
one = news()
two = persons()
three = pogoda()

print(one)
print(two)
print(three)
print(four)
print(seven)
print(five)
print(six)
