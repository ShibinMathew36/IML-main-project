import datetime
import glob
import string
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import collections as col

train_pos_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/train/pos/"
train_neg_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/train/neg/"
test_pos_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/test/pos/"
test_neg_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/test/neg/"
vectorizer_pos = CountVectorizer()
#vectorizer_neg = CountVectorizer()
pos_key_terms = []
key_term_occurance_pos = [[]]
key_term_occurance_neg = [[]]
context_positive = {}
context_negative = {}

# Constants used:
key_term_threshold = 2
context_span = 3

def get_data(path):
    data = []
    temp = 0
    for files in glob.glob(path + "*.txt"):
        infile = open(files)

        #fix case and remove punctuations, nunbers
        dat = infile.readline().lower()
        words = word_tokenize(dat.replace('<br />', ''))
        infile.close()
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        stripped = [w.translate(table) for w in words]

        # TODO: check efficiency will improve removing words smaller than 2 and greater than 15
        words = [word for word in stripped if word.isalpha()]

        # filter out stop words

        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        a = ' '.join(words)
        data.append(a)
        temp += 1
    return data


with Pool(4) as p:
    [train_positive, train_negative, test_positive, test_negative] = p.map(get_data, [train_pos_path, train_neg_path, test_pos_path, test_neg_path])

print("\n Data extracting and preprocessing done. Processed review count = ", str(4 * len(train_positive)))
print ("\n Training :")
X_train = vectorizer_pos.fit_transform(train_positive)
Y_train = vectorizer_pos.transform(train_negative) # should we fit it with same or different classifier objects ?
train_pos = X_train.toarray().sum(axis=0)
train_neg = Y_train.toarray().sum(axis=0)

print ("\n  Features extracted.")
print ("Total positive words = ", train_pos.sum())
print ("Total negative words = ", train_neg.sum()) # will this be a problem since i train with same classifier

pos_neg_ratio = float(train_neg.sum() / train_pos.sum())
pos_features = col.OrderedDict(zip(vectorizer_pos.get_feature_names(), train_pos))
neg_features = col.OrderedDict(zip(vectorizer_pos.get_feature_names(), train_neg))

print ("\n  Extracting key words.")
# TODO: include code to do dirichlet smoothing
temp = 0
np_Xtrain = np.array(X_train.toarray())
np_Ytrain = np.array(Y_train.toarray())
delta = 0
start = 0
for key in pos_features.keys():
    if (pos_features[key] / neg_features[key]) * pos_neg_ratio > key_term_threshold:
        pos_key_terms.append(key)
        start = datetime.datetime.now()
        key_term_occurance_pos.append(np.flatnonzero(np_Xtrain[:, temp]).tolist())
        delta = datetime.datetime.now() - start
            #add negative occurances too
    temp += 1
print (start)
print (str(len(pos_key_terms)), " positive key words extracted.")
"""
print ("\n  Extracting context terms.")
for index, key in enumerate(pos_key_terms):
    for review_index in key_term_occurance_pos[index]:
        review = train_positive[review_index].split()
        if key in review:
            position = review.index(key)
            start = max(0, position - context_span)
            end = min(len(review), (position + context_span + 1))
            subsring = review[start:position] + review[(position + 1):end]
            for word in subsring:
                if word in context_positive:
                    context_positive[word] += 1
                else:
                    context_positive[word] = 1
print (context_positive)


print ("\n Testing :")
X_test = vectorizer_pos.transform(test_positive)
print(X_test.toarray().sum(axis=0))
"""