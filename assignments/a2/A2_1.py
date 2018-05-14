import pandas as pd
pd_data = pd.read_csv('yelp_review.zip')
pd_data[:5]

all_data = list(zip(pd_data['text'], pd_data['stars']))
len(all_data)
all_data[:5]

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

from collections import Counter
train_count = Counter([rate_feature(rate) for review, rate in train_data])
devtest_count = Counter([rate_feature(rate) for review, rate in devtest_data])
test_count = Counter([rate_feature(rate) for review, rate in test_data])

print("Train Set: ", train_count)
print("Devtest Set: ",devtest_count)
print("Test Set: ", test_count)

##Ex2

text_train = [(review, rate_feature(rate)) for review, rate in train_data[:10000]]
text_devtest = [(review, rate_feature(rate)) for review, rate in devtest_data[:10000]]
text_test = [(review, rate_feature(rate)) for review, rate in test_data[:10000]]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(input='contents', stop_words='english', max_features=2000)
text_tfidf_train = tfidf.fit_transform([x for x, y in text_train])
text_tfidf_devtest = tfidf.transform([x for x, y in text_devtest])
text_tfidf_test = tfidf.transform([x for x, y in text_test])

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

sklearn_tfidfclassifier = MultinomialNB()
sklearn_tfidfclassifier.fit(text_tfidf_train, [y for x, y in text_train])

tfidf_Dev_pred = sklearn_tfidfclassifier.predict(text_tfidf_devtest)
tfidf_Test_pred = sklearn_tfidfclassifier.predict(text_tfidf_test)

f1_score(tfidf_Dev_pred, [y for x, y in text_devtest], average=None)
f1_score(tfidf_Test_pred, [y for x, y in text_test], average=None)

##Ex3

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(text_tfidf_train, [y for x, y in text_train])

logreg_Dev_pred = logreg.predict(text_tfidf_devtest)
logreg_Test_pred = logreg.predict(text_tfidf_test)

f1_score(logreg_Dev_pred, [y for x, y in text_devtest], average=None)
f1_score(logreg_Test_pred, [y for x, y in text_test], average=None)

##Ex5 
from sklearn.metrics import confusion_matrix
tn_dev_tfidf, fp_dev_tfidf, fn_dev_tfidf, tp_dev_tfidf = confusion_matrix(tfidf_Dev_pred, [y for x, y in text_devtest]).ravel()

tn_test_tfidf, fp_test_tfidf, fn_test_tfidf, tp_test_tfidf = confusion_matrix(tfidf_Test_pred, [y for x, y in text_test]).ravel()

tn_dev_logreg, fp_dev_logreg, fn_dev_logreg, tp_dev_logreg = confusion_matrix(logreg_Dev_pred, [y for x, y in text_devtest]).ravel()

tn_test_logreg, fp_test_logreg, fn_test_logreg, tp_test_logreg = confusion_matrix(logreg_Test_pred, [y for x, y in text_test]).ravel()





