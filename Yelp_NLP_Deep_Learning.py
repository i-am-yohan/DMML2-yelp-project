import pandas as pd
import numpy as np
import tensorflow as tf

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


In_Reviews = pd.read_csv('file:/home/hduser/DMML2/Project/ETL/Data/yelp_dataset/reviews_stratified.csv'
                                 , delimiter = '|'
                                 , quotechar = "'"
                                 , escapechar = '\\'
                                 #, nrows = 100
                                 )

#Encoding the variable to account for ordinal properties
In_Reviews['target_1'] = 0
In_Reviews['target_2'] = 0
In_Reviews['target_3'] = 0
In_Reviews['target_4'] = 0

In_Reviews.loc[In_Reviews.stars == 2.0,'target_1'] = 1
In_Reviews.loc[In_Reviews.stars == 3.0,'target_1'] = 1
In_Reviews.loc[In_Reviews.stars == 3.0,'target_2'] = 1
In_Reviews.loc[In_Reviews.stars == 4.0,'target_1'] = 1
In_Reviews.loc[In_Reviews.stars == 4.0,'target_2'] = 1
In_Reviews.loc[In_Reviews.stars == 4.0,'target_3'] = 1
In_Reviews.loc[In_Reviews.stars == 5.0,'target_1'] = 1
In_Reviews.loc[In_Reviews.stars == 5.0,'target_2'] = 1
In_Reviews.loc[In_Reviews.stars == 5.0,'target_3'] = 1
In_Reviews.loc[In_Reviews.stars == 5.0,'target_4'] = 1

#Change index
In_Reviews.index = In_Reviews['review_id']


#Apply spell checker
In_Reviews_ABT = In_Reviews[['stars','text','target_1','target_2','target_3','target_4']]
spell = Speller(lang = 'en')

#convert to spark because Pandas is slow!!
#conf = SparkConf().setMaster("local")
#sc = SparkContext(conf = conf)
#sqlContext = SQLContext(sc)
#spell_udf = udf(spell)

#In_Reviews_ABT_sp = sqlContext.createDataFrame(In_Reviews_ABT)
#In_Reviews_ABT_sp = In_Reviews_ABT_sp.withColumn('text' , spell_udf(col('text')))
#In_Reviews_ABT = In_Reviews_ABT_sp.toPandas()
#sc.stop()

#Get train and test split
In_Reviews_ABT['weight'] = 1
train = In_Reviews_ABT.sample(n=320000, random_state=198666, weights='weight')
test = In_Reviews_ABT[~In_Reviews_ABT.index.isin(train.index)]


# I'm using GLoVe word vectors to get pretrained word embeddings
embed_size = 200
max_features = 20000
maxlen = 200

embedding_file = '/home/hduser/DMML2/Project/NLP/glove.6B/glove.6B.200d.txt'

# read in embeddings
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))

#
class_names = ['target_1','target_2','target_3','target_4']
Y_train = train[class_names].values
Y_test = test[class_names].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train['text'].values))
X_train = tokenizer.texts_to_sequences(train['text'].values)
X_test = tokenizer.texts_to_sequences(test['text'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)


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
x = SpatialDropout1D(0.9)(x)
x = Bidirectional(LSTM(40, return_sequences=True))(x)
x = SpatialDropout1D(0.9)(x)
x = Bidirectional(GRU(40, return_sequences=True))(x)
x = SpatialDropout1D(0.9)(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(4, activation = 'sigmoid')(conc)

model = Model(inputs = inp, outputs = outp)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta=0, patience =3)
checkpoint = ModelCheckpoint(monitor = 'val_loss' , save_best_only = True, filepath='/home/hduser/DMML2/Project/NLP/yelp_best_model.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, Y_train, batch_size = 32, epochs = 20, validation_split=0.1
          )
model.save('/home/hduser/DMML2/Project/NLP/yelp_best_model2.hdf5')

local = tf.keras.models.load_model('/home/hduser/DMML2/Project/NLP/yelp_best_model2.hdf5')
y_predict = local.predict([x_test], batch_size=1024, verbose =1)

y_actual_stars = test['stars'].values
y_predict_stars = np.sum(np.round(y_predict), 1) + 1

print('Confusion Matrix')
print(metrics.confusion_matrix(y_actual_stars, y_predict_stars)/80000)

print('Accuracy Score')
print(metrics.accuracy_score(y_actual_stars, y_predict_stars))