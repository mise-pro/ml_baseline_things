import re
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import pymorphy2


def normalize_string2(txt):
    morph = pymorphy2.MorphAnalyzer()
    new_text=''
    txt = re.sub('[^\'а-яА-Я]',' ',txt)
    txt = txt.lower()
    grams_exclusion = {'PREP','CONJ','PRCL','INTJ','Name','Surn','Patr','Orgn','Trad'}
    words = txt.split()
    for word in words:
        #if word not in {'здравствуйте', 'добрый', 'доброе', 'вечер', 'утро', 'день', 'привет'}:
        p = morph.parse(word)[0]
        #print (p)
        if not any (tag in p.tag for tag in grams_exclusion) and (len(p) > 2 or lemma=='не'):
            new_text += ' ' + p.normal_form
    new_text = new_text [1:]
    return new_text

def normalize_string(txt):
    lem = WordNetLemmatizer()

    new_text=''
    txt = re.sub('[^\'а-яА-Я]',' ',txt)
    txt = txt.lower()
    # txt = re.sub('здравствуйте', ' ', txt)
    words = txt.split()
    for word in words:
        lemma = lem.lemmatize(word)
        if len(lemma) > 2 or lemma=='не':
            new_text += ' ' + lemma
    # new_text = new_text [:]
    return new_text
	
def get_review_details(fullReview, productId):# should be type of parser
    # fullReview = parser.find('div', attrs={'class':'n-product-review-item'})
    rating = 0
    error = 0
    negative = ""
    positive = ""
    comments = ""
    try:
        rating = int(fullReview.find('div', attrs={'class': 'rating__value'}).text)

        for reviewPhrase in fullReview.findAll('dl', attrs={'class': 'n-product-review-item__stat'}):
            if len(reviewPhrase.findAll('dt', attrs={'class': 'n-product-review-item__title'})) > 0:
                if 'достоинства' in (reviewPhrase.find('dt', attrs={'class': 'n-product-review-item__title'}).text).lower():
                    positive = reviewPhrase.find('dd', attrs={'class': 'n-product-review-item__text'}).text
                    # print(positive)
                if 'недостатки:' in (reviewPhrase.find('dt', attrs={'class': 'n-product-review-item__title'}).text).lower():
                    negative = reviewPhrase.find('dd', attrs={'class': 'n-product-review-item__text'}).text
                    # print(negative)
                if 'комментарий:' in (reviewPhrase.find('dt', attrs={'class': 'n-product-review-item__title'}).text).lower():
                    comments = reviewPhrase.find('dd', attrs={'class': 'n-product-review-item__text'}).text
                    # print(comments)
            else:
                comments = reviewPhrase.find('dd', attrs={'class': 'n-product-review-item__text'}).text

        return productId, rating, positive, negative, comments, error
    except:
        print('Error occuried while parsing product {}, skipping'.format(productId))
        error = 1
        raise ValueError('need debug here')
        return productId, rating, positive, negative, comments, error
# get_review_details(parser.find('div', attrs={'class':'n-product-review-item'}), 1023)

