import pandas as pd
pd_data = pd.read_csv('yelp_review.zip')
pd_data[:5]

all_data = list(zip(pd_data['text'], pd_data['stars']))
len(all_data)
all_data[:5]

from matplotlib import pyplot as plt

from collections import Counter
c = Counter([rating for text, rating in all_data])
c

plt.bar(range(1,6), [c[1], c[2], c[3], c[4], c[5]])

import random
random.seed(1234)
random.shuffle(all_data)
train_data, devtest_data, test_data = all_data[:500000], all_data[500000:510000], all_data[510000:520000]

##Ex1
def rate_feature(rate):
    if rate == 5:
        return 'it has 5 stars'
    else:
        return 'it does not have 5 stars'

train_count = Counter([rate_feature(rate) for review, rate in train_data])
devtest_count = Counter([rate_feature(rate) for review, rate in devtest_data])
test_count = Counter([rate_feature(rate) for review, rate in test_data])

train_count
devtest_count
test_count

##Ex2
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stopset = stopwords.words("english")
tokenizer = RegexpTokenizer(r'\w+')

train_set = [(rate_feature(rate), tokenizer.tokenize(review)) for review, rate in train_data]

devtest_set = [(rate_feature(rate), tokenizer.tokenize(review)) for review, rate in devtest_data]

test_set = [(rate_feature(rate), tokenizer.tokenize(review)) for review, rate in test_data]

a = []
temp = [[a.append(w.lower()) for w in review if w.lower not in stopset] for rate, review in train_set]
all_words = nltk.FreqDist(a)
word_features = list(all_words)[:2000]

def vector_features(words):
    "Return a vector of features for sklearn"
    words_lower = [w.lower() for w in words]
    result = []
    for w in word_features:
        if w in words_lower:
            result.append(1)
        else:
            result.append(0)
    return result
    
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

