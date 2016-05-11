#suppressing warnings
import warnings
warnings.filterwarnings("ignore")
#importing features
import json
import re
import numpy as np
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from numpy import average
from timeit import default_timer as timer

def pruneWords(line_words_preprocess):
    for word in line_words_preprocess:
        line_words_preprocess[line_words_preprocess.index(word)] = re.sub(r'([$\W])','',word.lower())
    return list(set(line_words_preprocess))

def bagOfwords(file_name):
    f = open(file_name)
    line = f.readline()
    bag_of_words = []
    i=0
    review_sparse_vect = []
    rating_sparse_vect = []

    # limiting no of disk operations based on requirement
    while line:
        #print line
        review = json.loads(line)

        # pruning with regular expressions to generate words
        line_words = pruneWords(list(set(review["text"].split())))

        # updating bag of words when encountering a new word
        for word in line_words:
            if word not in bag_of_words:
                bag_of_words.append(word)

        review_sparse_vect.append(list(np.zeros(len(bag_of_words))))
        rating_sparse_vect.append(review['stars'])

        for word in line_words:
            review_sparse_vect[i][bag_of_words.index(word)] = 1

        i=i+1
        if i >= samples:
            break
        line = f.readline()

    # increasing the size of vectors to bag of words
    for vect in review_sparse_vect:
        diff = len(bag_of_words) - len(vect)
        vect += [0]*diff

    f.close
    return review_sparse_vect,rating_sparse_vect

def classifier(file_name):
    review_sparse_vect,rating_sparse_vect = bagOfwords(file_name)
    # svm SVC -- Support vector classification One Vs rest or One Vs all
    clf = SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False, decision_function_shape= 'ovr')
    clf.fit(review_sparse_vect, rating_sparse_vect)
    #print("Fitting completed")
    predicted = cv.cross_val_predict(clf, review_sparse_vect,rating_sparse_vect, cv=10)
    #print (predicted)
    rating_sparse_vect = np.array(rating_sparse_vect)
    #print(abs(predicted - rating_sparse_vect)/4)
    # error calculation
    error  = average(abs(predicted - rating_sparse_vect)/4)
    print("Accuracy :   %d"%((1- error)*100))

file_name = 'D:\LargeDatasets\yelp\GoogleClassShare\yelp_academic_dataset_review_preprosessd.json'
for i in [100,500,1000,2000]:       #runing the code for various samples sizes to calculate the metrics for each sample size
    samples = i
    start = timer()
    print("The number of samples under process  :%d"%samples)
    bag_of_words = classifier(file_name)
    print ("Processing time for %d samples using 10 cross validation is %f"%(samples,timer() - start))
    print ("-"*70, "\n")
