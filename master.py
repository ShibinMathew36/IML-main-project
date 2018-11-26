import gc
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
vectorizer_neg = CountVectorizer()
key_term_occurance_pos = []
key_term_occurance_neg = []
context_terms = []


# Constants used:
# TODO: all values are dummy, need to figure out actuals
key_term_threshold = 5
context_span = 3
context_term_val_threshold = 0

def get_data(path):
    data = []
    temp = 0
    for files in glob.glob(path + "*.txt"):
        if temp == 100:
            break
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

def get_key_terms(X_train, Y_train,  prim_freq, sec_freq, prim_sec_ratio):
    # TODO: include code to do dirichlet smoothing
    prim_train = np.array(X_train.toarray())
    sec_train = np.array(Y_train.toarray())
    temp = 0
    prim_key_terms = col.OrderedDict()
    prim_term_occurance = []
    sec_term_occurance = []

    for key in prim_freq.keys():
        temp_sec_freq = 1
        flag = False
        if key in sec_freq.keys():
            temp_sec_freq = sec_freq[key]
            flag = True
        score = float((prim_freq[key] / temp_sec_freq) * prim_sec_ratio)
        if score > key_term_threshold:
            prim_key_terms[key] = score
            prim_term_occurance.append(np.flatnonzero(prim_train[:, temp]).tolist())
            if flag:
                sec_term_occurance.append(np.flatnonzero(sec_train[:, list(sec_freq.keys()).index(key)]).tolist())
            else:
                sec_term_occurance.append([])
        temp += 1
    return prim_key_terms, prim_term_occurance, sec_term_occurance


def get_context_dict(key_terms, key_term_occurance, train):
    context_dict = {}
    count = 0
    for index, key in enumerate(key_terms):
        for review_index in key_term_occurance[index]:
            review = train[review_index].split()
            if key in review:
                position = review.index(key)
                start = max(0, position - context_span)
                end = min(len(review), (position + context_span + 1))
                substring = review[start:position] + review[(position + 1):end]
                for word in substring:
                    if word in context_dict:
                        temp = (context_dict[word][0], context_dict[word][1] + 1)
                        context_dict[word] = temp
                    else:
                        context_dict[word] = (key, 1)
                    count += 1
    return context_dict, count


def get_context_terms(context_prim, context_sec, context_prim_count, context_sec_count):
    context_terms = {}
    active_prim_keys = set()
    for val in context_prim.keys():
        neg_freq = 0 # is the ideal initializer ?
        if val in context_sec:
            neg_freq = context_sec[val][1]
        #print ((context_prim[val] / context_prim_count) - (neg_freq / context_sec_count), end="  ")
        score = float(context_prim[val][1] / context_prim_count) - (neg_freq / context_sec_count)
        if score >= context_term_val_threshold:
            context_terms[val] = score
            active_prim_keys.add(context_prim[val][0])
    return context_terms, active_prim_keys

if __name__ == '__main__':

    p = Pool(4)
    [train_positive, train_negative, test_positive, test_negative] = \
        p.map(get_data, [train_pos_path, train_neg_path, test_pos_path, test_neg_path])
    gc.collect()
    print("\n Data extracting and preprocessing done. Processed review count = ", str(4 * len(train_positive)))
    print ("\n Training :")
    X_train = vectorizer_pos.fit_transform(train_positive)
    Y_train = vectorizer_neg.fit_transform(train_negative) # should we fit it with same or different classifier objects ?
    train_pos = X_train.toarray().sum(axis=0)
    train_neg = Y_train.toarray().sum(axis=0)
    print ("\n  Features extracted.")
    print ("Total positive words = ", train_pos.sum())
    print ("Total negative words = ", train_neg.sum()) # will this be a problem since i train with same classifier

    pos_features = col.OrderedDict(zip(vectorizer_pos.get_feature_names(), train_pos))
    neg_features = col.OrderedDict(zip(vectorizer_neg.get_feature_names(), train_neg))

    print ("\n  Extracting key words.")
    gc.collect()
    pos_key_terms, key_occurrance_pp, key_occurrance_pn = get_key_terms(X_train, Y_train, pos_features, neg_features, float(train_neg.sum() / train_pos.sum()))
    neg_key_terms, key_occurrance_nn, key_occurrance_np = get_key_terms(Y_train, X_train, neg_features, pos_features, float(train_pos.sum() / train_neg.sum()))
    print (str(len(pos_key_terms.keys())), " positive key words extracted.")
    #print (pos_key_terms)
    print (str(len(neg_key_terms.keys())), " negative key words extracted.")
    #print (neg_key_terms)
    print ("\n  Extracting context terms.")
    
    # TODO: parallelize this code, can go faster
    # extracting context term occurrences and count
    context_pp, context_pp_count = get_context_dict(list(pos_key_terms.keys()), key_occurrance_pp, train_positive)
    print (len(context_pp.keys()), context_pp_count)
    context_pn, context_pn_count = get_context_dict(list(pos_key_terms.keys()), key_occurrance_pn, train_negative)
    print (len(context_pn.keys()), context_pn_count)
    context_nn, context_nn_count = get_context_dict(list(neg_key_terms.keys()), key_occurrance_nn, train_negative)
    print (len(context_nn.keys()), context_nn_count)
    context_np, context_np_count = get_context_dict(list(neg_key_terms.keys()), key_occurrance_np, train_positive)
    print (len(context_np), context_np_count)
    key_occurrance_pp.clear(); key_occurrance_pn.clear(); key_occurrance_nn.clear(); key_occurrance_np.clear
    gc.collect()
    # extracting context terms
    pos_context_terms, active_pos_keys = get_context_terms(context_pp, context_pn, context_pp_count, context_nn_count)
    print ("\n \n")
    neg_context_terms, active_neg_keys = get_context_terms(context_nn, context_np, context_nn_count, context_pp_count)
    context_pp.clear(); context_pn.clear(); context_nn.clear(); context_np.clear()
    gc.collect()

    print ("Extracted ", str(len(pos_context_terms.keys())), " positive context terms.")
    #print (pos_context_terms)
    print ("Extracted ", str(len(neg_context_terms.keys())), " negative context terms.")
    #print (neg_context_terms)
    print ("\n Testing :")
    gc.collect()
    X_test = vectorizer_pos.transform(test_positive)
    print(X_test.toarray().sum(axis=0))
    
