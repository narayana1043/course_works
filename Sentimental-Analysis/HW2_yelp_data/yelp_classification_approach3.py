"""
1.Removing Stopwords, using sentence level pruning with word tokenizer and
    performing stemming

2.Improving performance with conda or theano by utilizing GPU --- work in
    progress hoping to complete soon -- unable to install theano reason - on
    windows to install theano there are other mandatory stuff that needs to
    handled with care
    time to look -- spring break
"""

# suppressing warnings
import warnings
import json
import nltk
import numpy as np
from timeit import default_timer as timer
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from numpy import average
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as PortStem

warnings.filterwarnings("ignore")
# importing features
# nltk.download('punkt')
# #downloading for punkt is used for part of speech tagging
# nltk.download('stopwords')
# #downloading stopwrods from english language
stop_words = set(stopwords.words("english"))
stemmer = PortStem()

start = timer()


def prune_words(review_text):
    review_sent_split = review_text.split('.')
    review_word_bag = []
    for sentence in review_sent_split:
        words = word_tokenize(sentence)
        review_word_bag += [stemmer.stem(word) for word in words if
                            word not in stop_words]
        # removing stopwords and stemming
    return review_word_bag


def vect_add(vecta, vectb):
    return vecta + vectb


def bag_of_words(file_name):
    f = open(file_name)
    line = f.readline()
    bag_of_words = []
    i = 0
    review_sparse_vect = []
    rating_sparse_vect = []

    # limiting no of disk operations based on requirement
    while line:
        # print line
        review = json.loads(line)

        # pruning with regular expressions to generate words
        review_word_bag = prune_words(review["text"])

        # updating bag of words when encountering a new word
        bag_of_words += [word for word in review_word_bag if
                         word not in bag_of_words]

        review_sparse_vect.append(list(np.zeros(len(bag_of_words))))
        rating_sparse_vect.append(review['stars'])

        for word in review_word_bag:
            review_sparse_vect[i][bag_of_words.index(word)] = 1

        i = i + 1
        if i >= samples:
            break
        line = f.readline()

    # increasing the size of vectors to bag of words
    for vect in review_sparse_vect:
        diff = len(bag_of_words) - len(vect)
        # vect = vectAdd(vect, np.zeros(len(diff)))
        vect += [0] * diff

    f.close
    return review_sparse_vect, rating_sparse_vect


def classifier(file_name):
    review_sparse_vect, rating_sparse_vect = bag_of_words(file_name)
    # svm SVC -- Support vector classification One Vs rest or One Vs all
    clf = SVC(C=1, kernel='linear', gamma=1, verbose=False, probability=False,
              decision_function_shape='ovr')
    clf.fit(review_sparse_vect, rating_sparse_vect)
    # print("Fitting completed")
    predicted = cv.cross_val_predict(clf, review_sparse_vect,
                                     rating_sparse_vect, cv=10)
    # print (predicted)
    rating_sparse_vect = np.array(rating_sparse_vect)
    error = average(abs(predicted - rating_sparse_vect) / 4)
    print("Accuracy :   %d" % ((1 - error) * 100))


file_name = 'D:\LargeDatasets\yelp\GoogleClassShare\
                yelp_academic_dataset_review_preprosessd.json'
for i in [100, 500, 1000, 2000]:
    # runing the code for various samples sizes to calculate the metrics for
    #  each sample size
    samples = i
    start = timer()
    print("The number of samples under process  :%d" % samples)
    bag_of_words = classifier(file_name)
    print("Processing time for %d samples using 10 cross validation is %f" % (
        samples, timer() - start))
    print("-" * 70, "\n")
