# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from gensim import models
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


Word2vecModel = models.Word2Vec.load("Word2vec_Vector")

class max_ensemble(object):

    def create_datavecs(self,sentences):
        num_features=300
        DataVecs = self.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        return DataVecs

    @staticmethod
    def getAvgFeatureVecs(reviews,word_vector_news, num_features):
        count = 0
        feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
        for sentence in reviews:
            feature_vecs[count] = max_ensemble.makeFeatureVec(sentence,word_vector_news,
                                                   num_features=300)
            count = count + 1
        return feature_vecs


    @staticmethod
    def makeFeatureVec(words, model,num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        # print sentence
        ehst_vec_set = set(model.wv.index2word)
        count = 0
        for word in words:
            if word in ehst_vec_set:
                count = count + 1
                vector_w2vec = model[word]
                feature_vec = np.add(feature_vec, vector_w2vec)

        if count == 0:
            feature_vec = np.ones((num_features,), dtype="float32")
        else:
            feature_vec = np.divide(feature_vec, count)
        return feature_vec

    @staticmethod
    def create_ngrams(text):
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        analyzer = vectorizer.build_analyzer()
        return analyzer(text)


