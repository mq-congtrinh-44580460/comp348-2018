import nltk
import re
from nltk.corpus import stopwords
import time
import sys
from multiprocessing import Pool
import collections
regex =re.compile('(RB.|VB.|JJ.|NN|NNS)$')


def process_review(review):
	stopset = stopwords.words("english")
	tokens = nltk.word_tokenize(review)
	tagged = nltk.pos_tag(tokens)
	result = []
	for word, tag in tagged: 
		if regex.match(tag) and word not in stopset:
			result.append(word.lower())
	return result


def proposed_process(start, end):
	result = []
	for review, rate in train_data[start:end]:
		temp = (process_review(review), rate_feature(rate))
		result.append(temp)
	return result


##proposed_train = [(process_review(review), rate_feature(rate)) for review, rate in train_data[:250000]]

p = Pool(8)
start = time.time()
proposed=[]
word=[]
proposed = (p.starmap(proposed_process, [(0, 100000),(100000, 200000)))

##Use 200000 train set

proposed_train = proposed[0] + proposed[1] + proposed[2]
end = time.time()
print(end-start)
proposed_devtest = [(process_review(review), rate_feature(rate)) for review, rate in devtest_data]
proposed_test = [(process_review(review), rate_feature(rate)) for review, rate in test_data]

counter = collections.Counter([w for review, rate in proposed_train
					for w in review])

top2000words = [w for (w,count) in counter.most_common(2000)]

def document_features(words):
	"Return the document features for an NLTK classifier"
	words_lower = [w.lower() for w in words]
	result = dict()
	for w in top2000words:
		result['has(%s)' % w] = (w in words_lower)
	return result

train_features = [(document_features(x), y) for x, y in train_data]
devtest_features = [(document_features(x), y) for x, y in proposed_devtest]
classifier = nltk.NaiveBayesClassifier.train(train_features)
