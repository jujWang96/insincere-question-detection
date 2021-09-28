import pandas as pd
import numpy as np
import re
import nltk 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#nltk.download('punkt')
stemmer = SnowballStemmer("english")
stoplist = stopwords.words('english')


def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text


contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


def addCapTag(word):
    """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
    if(len(re.findall("[A-Z]{3,}", word))):
        word = word.replace('\\', '' )
        transformed = re.sub("[A-Z]{3,}", "ALL_CAPS_"+word, word)
        return transformed
    else:
        return word


def tokenize(text):
    """
    apply stemming and add all capital tag, and transform to lower case 
    """
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        w = addCapTag(w)
        w = w.lower()
        w = stemmer.stem(w)
        finalTokens.append(w)
    text = " ".join(finalTokens)
    return text


def transform_text(df):
    """
    apply all the preprocessing operations to each row of df
    """
    
    for index, row in df.iterrows():
        row['question_text'] = removeNumbers(row['question_text'])
        row['question_text'] = replaceContraction(row['question_text'])
        row['question_text'] = tokenize(row['question_text'])

    return df


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


 
    
    











    
