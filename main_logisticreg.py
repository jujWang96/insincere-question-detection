# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
import nltk
from nltk.corpus import stopwords 
import string
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import helper
stemmer = SnowballStemmer("english")


def main():
  # read in data
  train = pd.read_csv('train.csv',nrows = 50000)
  nltk.download('stopwords')
  eng_stopwords = set(stopwords.words('english'))

  #creating new features 
  # Number of words
  train["num_words"] = train['question_text'].apply(lambda x: len(str(x).split()))

  # Number of unique words
  train["num_unique_words"] = train['question_text'].apply(lambda x: len(set(str(x).split())))

  # Number of characters
  train["num_chars"] = train['question_text'].apply(lambda x: len(str(x)))

  # Number of stopwords
  train['num_stopwords'] = train['question_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

  # Number of punctuations
  train["num_punctuations"] =train['question_text'].apply(lambda x: len([p for p in str(x) if p in string.punctuation]) )

  # Number of upper case words
  train['num_words_upper'] = train['question_text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

  # Number of title case words
  train["num_words_title"] = train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

  # Average length of words
  train["mean_word_len"] = train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

  train_trans = helper.transform_text(train)
    
  ## split to train and val
  train_df, val_df = train_test_split(train_trans, test_size=0.3, random_state=1)

  #apply TFIDF transformation 
  nltk.download('stopwords')
  eng_stopwords = set(stopwords.words('english'))

  analyzer = TfidfVectorizer().build_analyzer()
  def stemmed_words(doc): return (stemmer.stem(w) for w in analyzer(doc))

  tfidf_vectorizer=TfidfVectorizer(analyzer=stemmed_words,max_features = 5000,max_df = 0.5)
  tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(train_df.loc[:,"question_text"])
  Tfidf_train_df = pd.DataFrame(tfidf_vectorizer_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names())
  Tfidf_val_df =  pd.DataFrame(tfidf_vectorizer.transform(val_df.loc[:,"question_text"]).toarray(),columns=tfidf_vectorizer.get_feature_names())

  Tfidf_train = Tfidf_train_df.to_numpy()
  Tfidf_val = Tfidf_val_df.to_numpy()


  new_feature_train = train.loc[Tfidf_train_df.index.values,["num_words","num_unique_words","num_punctuations","num_words_upper","num_words_title","mean_word_len"]].to_numpy()
  new_feature_val = train.loc[Tfidf_val_df.index.values,["num_words","num_unique_words","num_punctuations","num_words_upper","num_words_title","mean_word_len"]].to_numpy()

  train_X = np.concatenate((Tfidf_train, new_feature_train),axis=1)
  val_X =  np.concatenate((Tfidf_val, new_feature_val),axis=1)
  train_y = train_df['target']
  val_y = val_df['target']
  
  C = np.array([0.1,1,10,50,100])
  C_score = np.zeros(len(C))
  C_f1score = np.zeros(len(C))
  for c,i in zip(C,np.arange(len(C))):
    lreg = LogisticRegression(C=c, solver='liblinear',penalty='l1').fit(train_X, train_y)
    C_score[i] = lreg.score(val_X,val_y)
    C_f1score[i] = f1_score(lreg.predict(val_X),val_y)
  print("The F1 score for Logistic Regression with C = {}, is {}".format(C, C_f1score))




if __name__=="__main__":
  main()
