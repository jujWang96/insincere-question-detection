import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
import nltk
from nltk.corpus import stopwords 
import string
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers
import os
import helper



def plot_metrics(history,outfilename):
    """
    plot the training history 
    """
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend() 
    fig = plt.figure()
    plt.savefig(outfilename+"_metric.fig")
    plt.close(fig)


def init_bias(arr):
    """
    initialize bias term for imbalanced dataset 
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    b = np.log([pos/neg])

    return b

def class_w(arr):
    """
     calculate different class weights for trainning unbalanced data
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


def prepare_embedding_mat(max_features)
    """
    create embedding matrix with pretrained embedding
    """

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


def make_model(embedding_vecor_length,  output_bias,embedding_matrix ,max_features = 50000):
    """
    build a model with non-pretrained embedding 
    """    
    max_review_length=300
    model = Sequential()
    model.add(Embedding(max_features, embedding_vecor_length, input_length=max_review_length))
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
    ## some config values
    embed_size = 100 # how big is each word vector
    max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 300 # max number of words in a question to use
    BATCH_SIZE = 256
    train = pd.read_csv('train.csv')
    print("Train shape:", train.shape)    
    ##apply preprocessing operations to both train and test data
    train_trans = helper.transform_text(train)
    
    ## split to train and val
    train_df, val_df = train_test_split(train_trans, test_size=0.3, random_state=1)

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

    embedding_mat = prepare_embedding_mat(max_features,tokenizer)
    model = make_model(embed_size,initial_bias,embedding_mat)
    early_stopping = early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
        )
    
    history = model.fit(
        train_X,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=10,
        callbacks = [early_stopping]
        validation_data=(val_X, val_y)
        class_weight = class_weight)
    
    ##predict for test data and evaluate the performance on various metrics 
    pred_y = model.predict_classes(val_X, batch_size=BATCH_SIZE, verbose=0)
    pred_results = model.evaluate(val_X, val_y,
                                batch_size=BATCH_SIZE,
                                  verbose=0)
    plot_metrics(history,'fasttext')
    for name, value in zip(model.metrics_names, pred_results):
        print(name, ': ', value)
    print()
    
if name=="__main__":
    main()
