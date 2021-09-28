import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import nltk
from nltk.corpus import stopwords 
import string
import time
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Bidirectional, GlobalMaxPool1D
import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers
import os
import helper



def init_bias(arr):
    """
    initialize bias term for imbalanced dataset 
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    b = np.log([pos/neg])

    return b[0]

def class_w(arr):
    """
     calculate different class weights for trainning unbalanced data
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    total = pos+neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight

def make_model(embedding_vecor_length,  output_bias ,top_words = 50000):
    """
    build a model with non-pretrained embedding 
    """    
    max_review_length=300
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Bidirectional(LSTM(20,dropout=0.3, recurrent_dropout=0.2,return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(10,activation = 'relu'))
    model.add(Dense(1, activation='sigmoid',
                     bias_initializer = output_bias))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 
    
    
def main():
    ##load data
  
    embed_size = 100 # how big is each word vector
    max_features = 80000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 300 # max number of words in a question to use
    BATCH_SIZE = 256
    train = pd.read_csv('train.csv')
    print("Train shape:", train.shape)    
    ##apply preprocessing operations to training dataset
    train_trans = helper.transform_text(train)
    
    ## split to train and val
    
    train_size = len(train_trans.index)
    train_df, test = train_test_split(train_trans[:train_size], test_size=0.2, random_state=1)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=1)


    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_na_").values
    val_X = val_df["question_text"].fillna("_na_").values
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values
    
    ##initialize bias for unbalanced data 
    initial_bias = init_bias(train_y)
    class_weight = class_w(train_y)
    model = make_model(embed_size,initial_bias)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
        )
    
    history = model.fit(
        train_X,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=10,
        callbacks = [early_stopping],
        validation_data=(val_X, val_y),
        calss_weight = class_weight)

    ##predict for test data and evaluate the performance on various metrics 
    pred_y = model.predict_classes([val_X], batch_size=BATCH_SIZE, verbose=0)
    pred_results = model.evaluate([val_X], val_y,
                                batch_size=BATCH_SIZE,
                                  verbose=0)
    helper.plot_metrics(history,'non-pretrainedembedding')
    for name, value in zip(model.metrics_names, pred_results):
        print(name, ': ', value)
    print()
    
if __name__=="__main__":
    main()
