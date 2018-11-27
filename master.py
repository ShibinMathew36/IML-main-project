import gc
import glob
import string
import statistics as stat
import numpy as np
import collections as col
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

train_pos_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/train/pos/"
train_neg_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/train/neg/"
test_pos_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/test/pos/"
test_neg_path = "/home/jarvis/Documents/iml/Project/DataSets/Imdb_dataset/test/neg/"
vectorizer_pos = CountVectorizer()
vectorizer_neg = CountVectorizer()
key_term_occurance_pos = []
key_term_occurance_neg = []

ignore_stops = set(["mightn't", "shan't", "don't", 'isn', 'against', 'more', "wasn't", 'no', 'wasn', "weren't", "won't", 'mustn', 'shouldn', 'hadn', 'didn', 'doesn', "should've", 'very', "doesn't", 'needn', "didn't", 'wouldn', "needn't", 'below', "hasn't", "haven't", 'not', "wouldn't", 'over', "mustn't", 'mightn', 'hasn', "hadn't", "aren't", 'ain', "couldn't", 'haven', "isn't", 'don', 'few', 'weren', 'nor', 'does', 'couldn', 'but', 'down', "shouldn't", 'aren', 'won', "mightn't", "shan't", "don't", 'isn', 'against', 'more', "wasn't", 'no', 'wasn', "weren't", "won't", 'too', 'mustn', 'shouldn', 'hadn', 'didn', 'doesn', "should've", "doesn't", 'needn', 'shan', "didn't", 'wouldn', "needn't", 'below', "hasn't", "haven't", 'not', "wouldn't", 'over', 'most', "mustn't", 'mightn', 'above', 'hasn', "hadn't", "aren't", 'ain', "couldn't", 'haven', "isn't", 'don', 'off', 'couldn', "shouldn't", 'aren', 'won'])

stp_words = set(stopwords.words('english')) - ignore_stops
# training data * feature values
# feature_list: no_of_keys, max_key_score, min_key_score, avg_key_scores, std_dev_key_scores, inactive_keys, inactive_key_%
#               max_dist_btw_keys, min_distance_btw_keys
pos_feature_vals_train = []
neg_feature_vals_train = []

# Constants used:
key_term_threshold = 5
context_span = 3
context_term_val_threshold = 0

# returns extracted and cleaned reviews from respective path
def get_data(path):
    data = []
    temp = 0
    for files in glob.glob(path + "*.txt"):
        if temp == 1000:
            break
        infile = open(files)

        #fix case and remove punctuations, nunbers
        dat = infile.readline().lower()
        words = word_tokenize(dat.replace('<br />', ''))
        infile.close()
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        stripped = [w.translate(table) for w in words]
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        words = [w for w in words if not w in stp_words]
        a = ' '.join(words)
        data.append(a)
        temp += 1
    return data


# returns the key terms and their occurrences in both positive and negative reviews(helps extracting contexts)
def get_key_terms(X_train,  prim_freq, sec_freq, prim_sec_ratio):
    # TODO: include code to do dirichlet smoothing
    prim_train = np.array(X_train.toarray())
    temp = 0
    prim_key_terms = col.OrderedDict()
    prim_term_occurance = {}

    for key in prim_freq.keys():
        temp_sec_freq = 1
        if key in sec_freq:
            temp_sec_freq = sec_freq[key]
        score = (prim_freq[key] / temp_sec_freq) * prim_sec_ratio
        if score > key_term_threshold:
            prim_key_terms[key] = score
            prim_term_occurance[key] = np.flatnonzero(prim_train[:, temp]).tolist()
        temp += 1
    return prim_key_terms, prim_term_occurance

# fetches the substrings 3 words left and right of each key
def fetch_substring(review, key, position = -1):
    if position == -1:
        position = review.index(key)
    start = max(0, position - context_span)
    end = min(len(review), (position + context_span + 1))
    return review[start:position] + review[(position + 1):end]


#returns dict containing the associated key term and number of occurrences of each context term and total count
def get_context_dict(key_terms, key_term_occurance, train):
    context_dict = {}  # dict mapping each context term to associated key word and its frequency
    for key in key_terms:
        context_per_key = {}
        for review_index in key_term_occurance[key]:
            review = train[review_index].split()
            if key in review:
                substring = fetch_substring(review, key)
                for word in substring:
                    if word in context_per_key:
                        context_per_key[word] += 1
                    else:
                        context_per_key[word] = 1
        context_dict[key] = context_per_key 
    return context_dict


# extracts context terms based on score
def get_context_terms(key_context_prim, key_context_sec, prim_key_freq, sec_key_freq):
    key_context_score_map = {}
    for prim_kt, prim_kt_val in key_context_prim.items():
        prim_key_val = prim_key_freq[prim_kt]
        sec_kt_val = {}
        context_score_map = {}
        #TODO : fix dirichlet smoothing here
        neg_key_freq = 1
        if prim_kt in sec_key_freq:
            neg_key_freq = sec_key_freq[prim_kt]
        if prim_kt in key_context_sec:
            sec_kt_val = key_context_sec[prim_kt]
        for prim_context in prim_kt_val.keys():
            sec_context = 0
            if prim_context in sec_kt_val:
                 sec_context = sec_kt_val[prim_context]
            score = float(prim_kt_val[prim_context]/ prim_key_val) - float(sec_context / neg_key_freq)
            if score >= context_term_val_threshold:
                context_score_map[prim_context] = score
        key_context_score_map[prim_kt] = context_score_map
    return key_context_score_map


# returns max, avg and std dev of the distance between successive key terms in the review
def max_avg_stddev(values):
    if len(values) == 1:
        return [values[0]] * 3
    else:
        dif = []
        for index, val in enumerate(values):
            if index == (len(values) - 1):
                break
            dif.append(values[index + 1] - values[index])
        std_dev = 0
        if len(dif) > 1:
            std_dev = stat.stdev(dif)
        return [max(dif), float(sum(dif) / len(dif)), std_dev]


# returns the sliding window max for window sizes 10, 20, 30 
def sliding_window(interval_10):
    max_10 = 0
    max_20 = 0; temp_20 = 0
    max_30 = 0; temp_30 = 0
    for idx, val in enumerate(interval_10):
        temp_20 += val; temp_30 += val
        max_10 = max(max_10, val)
        if ((idx + 1) % 2 == 0):
           max_20 = max(temp_20, max_20)
           temp_20 = 0
        if ((idx + 1) % 3 == 0):
           max_30 = max(temp_30, max_30)
           temp_30 = 0
        if idx == (len(interval_10) - 1):
           max_20 = max(temp_20, max_20)
           max_30 = max(temp_30, max_30)
    return [max_10, max_20, max_30]


# extracts features related to context terms
def context_related_features(review, key_terms, active_keys, context_stored_scores):
    freqs = []
    percents = []
    common_context_scores = []
    context_score_ratio = []
    for key in key_terms.keys():
        contexts = context_stored_scores[key]
        if key in active_keys:
            context_score_sum = []
            for occurrence in key_terms[key]:
                substring = fetch_substring(review, key, occurrence)
                for l_string in substring:
                    if l_string in contexts:
                        context_score_sum.append(contexts[l_string]) # TODO:  is this across all contexts ?
                commons = set(substring).intersection(set(list(contexts.keys()))) 
                percents.append(float((len(commons) * 100) / len(contexts.keys())))
                freqs.append(len(commons))
                avg = []
                for context in commons:
                    avg.append(contexts[context])
                temp = float(sum(avg) / max(1, len(avg)))
                common_context_scores.append(temp)
                context_score_ratio.append(temp / max(1, (sum(context_score_sum) / max(1, len(context_score_sum)))))
        else:
            freqs.append(0)
            percents.append(0)
            common_context_scores.append(0)
            context_score_ratio.append(0)
    stddev_1 = 0; stddev_2 = 0; stddev_3 = 0; stddev_4 = 0
    if len(freqs) > 1:
        stddev_1 = stat.stdev(freqs)
        stddev_2 = stat.stdev(percents)
        stddev_3 = stat.stdev(common_context_scores)
        stddev_4 = stat.stdev(context_score_ratio)
        return [0]*12
    context_features = [max(freqs), float(sum(freqs) / max(1, len(freqs))), stddev_1]    
    context_features += [max(percents), float(sum(percents) / max(1, len(percents))), stddev_2]
    context_features += [max(common_context_scores), float(sum(common_context_scores) / max(1, len(common_context_scores))), stddev_3]
    context_features += [max(context_score_ratio), float(sum(context_score_ratio) / max(1, len(context_score_ratio))), stddev_4]
    return context_features


# extracts the first 23 feature values
def get_features_1_to_23(review, key_term_scores, word_frequencies, total_words, key_context_scores):
    active_keys = [kt for kt in key_context_scores if key_context_scores[kt]] # not sure if this is correct
    features = [0]*7
    keys_inter = {}
    key_inter_pos = []
    key_scores = []
    key_language_model = []
    interval_10 = []; temp = 0
    total_count = 0
    for index, word in enumerate(review):
        if word in key_term_scores:
            total_count += 1
            temp += 1
            if word in keys_inter:
                keys_inter[word] += [index]
            else:
                keys_inter[word] = [index]
            key_inter_pos.append(index)
            key_scores.append(key_term_scores[word])
            key_language_model.append(word_frequencies[word] / total_words)
        if (index + 1) % 10 == 0:
            interval_10.append(temp)
            temp = 0
        elif index == (len(review) - 1):
            interval_10.append(temp)
    inactive_keys = set(keys_inter.keys()).difference(set(active_keys))
    if total_count != 0:
        features[0] = total_count
        features[1] = max(key_scores)
        features[2] = min(key_scores)
        features[3] = float(sum(key_scores) / features[0])
        if features[0] == 1:
           features[4] = 0
        else:    
           features[4] = stat.stdev(key_scores)
        features[5] = len(inactive_keys)
        features[6] = float((features[5] * 100)/ total_count)
        features += max_avg_stddev(key_inter_pos)
        features += sliding_window(interval_10)
        features.append(max(list(key_term_scores.values()))) # TODO: check if this feature is correct
        features.append(float(total_count / len(key_term_scores.keys())))
        if len(key_term_scores.keys()) == 1:
            features.append(0)
        else:
            features.append(stat.stdev(list(key_term_scores.values())))
        features.append(float(sum(key_language_model) / len(key_language_model)))
        if len(key_language_model) == 1:
            features.append(0)
        else:
            features.append(stat.stdev(key_language_model))
        features += context_related_features(review, keys_inter, active_keys, key_context_scores)
    else:
        features = [0]*23 # update to total number of features
    return features


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
    print ("\n  Extracting Language model.")
    print ("Total positive words = ", train_pos.sum())
    print ("Total negative words = ", train_neg.sum()) # will this be a problem since i train with same classifier

    pos_word_freq = col.OrderedDict(zip(vectorizer_pos.get_feature_names(), train_pos))
    neg_word_freq = col.OrderedDict(zip(vectorizer_neg.get_feature_names(), train_neg))

    print ("\n  Extracting key words.")
    gc.collect()
    train_neg_sum = train_neg.sum()
    train_pos_sum = train_pos.sum()
    pos_key_scores, key_occurrance_pos = get_key_terms(X_train, pos_word_freq, neg_word_freq, float(train_neg_sum / train_pos_sum))
    neg_key_scores, key_occurrance_neg = get_key_terms(Y_train, neg_word_freq, pos_word_freq, float(train_pos_sum / train_neg_sum))
    print (str(len(pos_key_scores.keys())), " positive key words extracted.")
    print (str(len(neg_key_scores.keys())), " negative key words extracted.")

    print ("\n  Extracting context terms.")
    # extracting context term occurrences and count
    pos_key_context_map_temp = get_context_dict(pos_key_scores.keys(), key_occurrance_pos, train_positive)
    neg_key_context_map_temp = get_context_dict(neg_key_scores.keys(), key_occurrance_neg, train_negative)
    key_occurrance_pos.clear(); key_occurrance_neg.clear()
    gc.collect()

    # extracting context terms, active_pos_keys represents those positive keys with context terms associated with it
    pos_key_context_map = get_context_terms(pos_key_context_map_temp, neg_key_context_map_temp, pos_word_freq, neg_word_freq)
    print ("\n \n")
    # active_neg_keys represents those negative keys with context terms associated with it
    neg_key_context_map = get_context_terms(neg_key_context_map_temp, pos_key_context_map_temp, neg_word_freq, pos_word_freq)
    pos_key_context_map_temp.clear(); neg_key_context_map_temp.clear()
    gc.collect()

    print ("\n Extracting Features:")
    # TODO: vectorize here ?
    for rev_idx, review in enumerate(train_positive):
        pos_feature_vals_train.append(get_features_1_to_23(review.split(), pos_key_scores, pos_word_freq, train_pos_sum, pos_key_context_map))
           
    for rev_idx, review in enumerate(train_negative):   
        neg_feature_vals_train.append(get_features_1_to_23(review.split(), neg_key_scores,  neg_word_freq, train_neg_sum, neg_key_context_map))
    #print (pos_feature_vals_train)
    #print ("\n \n")
    #print (neg_feature_vals_train)
    print ("\n Testing :")
    # X_test = vectorizer_pos.transform(test_positive)
    # print(X_test.toarray().sum(axis=0))

