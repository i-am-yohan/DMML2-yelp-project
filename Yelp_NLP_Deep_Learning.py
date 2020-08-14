import pandas as pd
import numpy as np
import tensorflow as tf
import subprocess

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from autocorrect import Speller
from sklearn import metrics


from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import udf, col

#Important!! Edit these parameters here!!
in_Loc =  #the csv file input
embedding_file =  #The input text embeddings
out_model =  #The output model location

In_Reviews = pd.read_csv(in_Loc
                         , delimiter = ','
                         , quotechar = "'"
                         , escapechar = '\\'
                         #, nrows = 100
                         )

#Encoding the variable to account for ordinal properties
In_Reviews['stars'] = In_Reviews['stars'].apply(int)
In_Reviews = pd.get_dummies(In_Reviews , columns = ['stars'])

#Change index
In_Reviews.index = In_Reviews['review_id']


#Get ABT
In_Reviews_ABT = In_Reviews[['text','stars_1','stars_2','stars_3','stars_4','stars_5']]
spell = Speller(lang = 'en')


#Get train and test split
In_Reviews_ABT['weight'] = 1
train = In_Reviews_ABT.sample(n=320000, random_state=198666, weights='weight')
test = In_Reviews_ABT[~In_Reviews_ABT.index.isin(train.index)]


# I'm using GLoVe word vectors to get pretrained word embeddings
embed_size = 100
max_features = 10000
maxlen = 100


# read in embeddings
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))

#Get the Y data into vectors
class_names = ['stars_1','stars_2','stars_3','stars_4','stars_5']
Y_train = train[class_names].values
Y_test = test[class_names].values

#Embed the text to vectors
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train['text'].values))
X_train = tokenizer.texts_to_sequences(train['text'].values)
X_test = tokenizer.texts_to_sequences(test['text'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)


#Apply missing words to test set
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
missed = []
for word, i in word_index.items():
    if i >= max_features: break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        missed.append(word)


#Lets train the model!!!
inp = Input(shape = (maxlen,))
x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.5)(x)
#x = Conv1D(128, 5)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
avg_pool = GlobalAveragePooling1D()(x)
#max_pool = GlobalMaxPooling1D()(x)
#conc = concatenate([avg_pool, max_pool])
outp = Dense(5, activation = 'sigmoid')(avg_pool)


model = Model(inputs = inp, outputs = outp)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0, patience =3)
checkpoint = ModelCheckpoint(monitor = 'val_loss' , save_best_only = True, filepath=out_model)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, Y_train, batch_size = 32, epochs = 10, validation_split=0.1
          )

#Save the output model
model.save(out_model)
