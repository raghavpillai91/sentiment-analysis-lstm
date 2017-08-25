# -*- coding: utf-8 -*-
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from data_cleaning import CleanData
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

clean_obj = CleanData()
gender_file = "sentiment_tweets.csv"

def read_data(training_file):
    train = pd.read_csv(training_file,sep=',')
    train.dropna(axis=0,how='any')
    return train

def word_vec_train(sentences):
    model = Word2Vec(sentences=sentences,size=300,min_count=1,workers=8)
    model.init_sims(replace=True)
    model.save('Word2vec_Vector')

def create_ngrams(text):
    vectorizer = CountVectorizer(ngram_range=(1,2))
    analyzer = vectorizer.build_analyzer()
    return analyzer(text)


gender = read_data(gender_file)

sentences = gender['tweet']
print len(sentences)

total_sentences = []
total_tfidf_sentences = []
count = 0
for sentence in sentences:
    count += 1
    sen = ' '.join(clean_obj.clean_article(sentence))
    total_sentences.append(create_ngrams(sen))
    total_tfidf_sentences.append(' '.join(clean_obj.clean_article(sentence)))
    print count

word_vec_train(total_sentences)



