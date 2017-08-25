import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_cleaning import CleanData
from max_ensemble import max_ensemble
import pandas as pd
from gensim import models

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 1000

clean_obj = CleanData()
max_obj = max_ensemble()
Word2vecModel = models.Word2Vec.load("Word2vec_Vector")
training_file = 'sentiment_tweets.csv'
def read_data(training_file):
    train = pd.read_csv(training_file,sep=',')
    return train
# print 'read training file'
train = read_data(training_file=training_file)
clean_sentences = []
for sentence in train['tweet']:
    sen = ' '.join(clean_obj.clean_article(sentence))
    clean_sentences.append(max_obj.create_ngrams(sen))

train_class = train['sentiment']

num_features=300
DataVecs = max_obj.getAvgFeatureVecs(clean_sentences, Word2vecModel, num_features)

X_train, X_test, y_train, y_test = train_test_split(DataVecs, train_class, test_size=0.4, random_state=0)

# # truncate and pad input sequences
max_review_length = 300
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))