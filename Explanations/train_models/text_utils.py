import numpy as np
import re
import string as string
from tensorflow.keras.datasets import imdb  # use the same words index of the IMDB dataset
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import read_csv
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from glove_utils import pad_sequences as glove_pad_sequence

def normalize(x):
    for i in range(x.shape[1]):
        if x[:,i].ptp() == 0.:
            continue
        x[:,i] = (x[:,i] - x[:,i].min()) / x[:,i].ptp()
    return x
    
def clean_text(text, token='N'):
    text = text.lower()  # lower case words
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

def stem_lem(words):
    stemmatizer, lemmatizer = PorterStemmer(), WordNetLemmatizer()
    words = [stemmatizer.stem(w) for w in words]  # stemming
    words = [lemmatizer.lemmatize(w) for w in words]  # lemmization
    return words

def pad_sequences(inputs, pad_token, maxlen):
    """
    Pad a list of texts (encoded as list of indices, one for each word) up to a maxlen parameter.
    """
    X = []
    for input_ in inputs:
        X.append([])
        if len(input_) > maxlen:
            X[-1] = input_[:maxlen]
        else:
            X[-1] = input_ + [pad_token for _ in range(maxlen-len(input_))]
    return X

def imdb2indices(inputs):
    """
    Turn a list of texts (encoded as list of words) into indices, according to the words
     that are present in the imdb dataset (as implemented by Keras).
    """
    X = []  # results
    word2index = imdb.get_word_index()
    word2index = {k:(v+3) for k,v in word2index.items()}
    word2index["<PAD>"], word2index["<START>"], word2index["<UNK>"], word2index["<UNUSED>"] = 0,1,2,3
    for input_ in inputs:
        X.append([])
        for word in input_:
            idx = word2index.get(word, word2index["<UNK>"])
            X[-1].append(idx)
    return X

def imdb2text(x, reverse_index):
    """
    Return a review from an IMDB input expressed by indices.
    """
    decoded = [reverse_index.get(i - 3, "#") for i in x]
    return decoded

def test_SST(model, index2embedding, word2index, input_shape, maxlen, emb_dims,  data_path='./../data/SST_2'):
    # Load STT dataset (eliminate punctuation, add padding etc.)
    X_test = read_csv(data_path + '/eval/SST_2__TEST.csv', sep=',',header=None).values
    y_test = []
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s.strip()=='negative' else 1))
    X_test = X_test[:,0]
    n = -1  # you may want to take just some samples (-1 to take them all)
    X_test = X_test[:n]
    y_test = y_test[:n]
    # Inputs as Numpy arrays
    X_test = np.array([np.array(x) for x in X_test]) 
    X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
    X_test = np.asarray(glove_pad_sequence(X_test, maxlen=maxlen, emb_size=emb_dims))
    X_test = X_test.reshape(len(X_test), *input_shape)
    y_test = to_categorical(y_test, num_classes=2)
    res = model.evaluate(X_test, y_test)
    return res

def test_Twitter(model, index2embedding, word2index, input_shape, maxlen, emb_dims,  data_path='./../data/Twitter'):
    # Load Twitter dataset (eliminate punctuation, add padding etc.)
    X_test = read_csv(data_path + '/test.csv', sep=',',header=None).values
    y_test = []
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s==0 else 1))
    X_test = X_test[:,0]
    n = -1  # you may want to take just some samples (-1 to take them all)
    X_test = X_test[:n]
    y_test = y_test[:n]
    # Inputs as Numpy arrays
    X_test = np.array([np.array(x) for x in X_test]) 
    X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
    X_test = np.asarray(glove_pad_sequence(X_test, maxlen=maxlen, emb_size=emb_dims))
    X_test = X_test.reshape(len(X_test), *input_shape)
    y_test = to_categorical(y_test, num_classes=2)
    res = model.evaluate(X_test, y_test)
    return res

def test_IMDB(model, index2embedding, word2index, input_shape, maxlen, emb_dims, vocabulary_size=90000):
    # Load IMDB dataset 
    (_, _), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    # Split dataset into chunks of 10k each to prevent memory errors
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()]) 
    X_test = [imdb2text(x, reverse_index) for x in X_test]
    # Prepare test set in advance
    X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
    X_test = np.asarray(glove_pad_sequence(X_test, maxlen=maxlen, emb_size=emb_dims))
    X_test = X_test.reshape(len(X_test), *input_shape)
    y_test = to_categorical(y_test, num_classes=2)
    res = model.evaluate(X_test, y_test)
    return res    
    