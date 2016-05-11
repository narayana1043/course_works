'''
1. Built Support Vector Classifier (one Vs rest)
2. using 10 Cross Validation
3. Metrics used: classification report, confusion matrix, accuracy score, precision score, recall score

'''

#suppressing warnings
import warnings
warnings.filterwarnings("ignore")
#importing features
import json
import re
import numpy as np
from sklearn import cross_validation as cv
from sklearn.metrics import classification_report as cr, confusion_matrix as cm, accuracy_score as acc_score, precision_score as pre_score, recall_score as rc_score
from sklearn.svm import SVC
from timeit import default_timer as timer

#pruning the review using regular expressions inorder to split the review into a list of words
def pruneWords(line_words_preprocess):
    for word in line_words_preprocess:
        line_words_preprocess[line_words_preprocess.index(word)] = re.sub(r'([$\W])','',word.lower())
    return list(set(line_words_preprocess))

#building vector space model for every review text using bag of words
def bagOfwords(file_name):
    f = open(file_name)
    line = f.readline()
    bag_of_words = []
    i=0
    review_sparse_vect = []
    rating_sparse_vect = []
    while line:

        review = json.loads(line)

        # print ("---")
        # print (review["review_id"])
        # print (review["text"])
        # print (review["pos"])
        # print (review["stars"])
        # print (type(review))
        # print (review)
        # print ("---")

        line_words = pruneWords(list(set(review["text"].split())))

        for word in line_words:
            if word not in bag_of_words:
                bag_of_words.append(word)

        review_sparse_vect.append(list(np.zeros(len(bag_of_words))))

        if review['stars'] == 4 or review['stars'] == 5:
            rating_sparse_vect.append(1)
        else:
            rating_sparse_vect.append(0)

        for word in line_words:
            review_sparse_vect[i][bag_of_words.index(word)] = 1

        i=i+1
        if i >= samples:
            break
        line = f.readline()
    #increasing the length of all vector models to the length of final bag of words
    for vect in review_sparse_vect:
        diff = len(bag_of_words) - len(vect)
        vect += [0]*diff

    f.close
    return review_sparse_vect,rating_sparse_vect

def classifier(file_name):
    review_sparse_vect,rating_sparse_vect = bagOfwords(file_name)
    # support vector classifier one vs all
    clf = SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False, decision_function_shape= 'ovr')
    clf.fit(review_sparse_vect, rating_sparse_vect)
    #Model fitting completeion
    #print("Fitting completed")
    predicted = cv.cross_val_predict(clf, review_sparse_vect,rating_sparse_vect, cv=10)
    # calculation of metrics
    print ("accuracy_score\t", acc_score(rating_sparse_vect,predicted))
    print ("precision_score\t", pre_score(rating_sparse_vect, predicted))
    print ("recall_score\t", rc_score(rating_sparse_vect, predicted))
    print ("\nclassification_report:\n\n", cr(rating_sparse_vect,predicted))
    print ("\nconfusion_matrix:\n", cm(rating_sparse_vect,predicted))

file_name = 'D:\LargeDatasets\yelp\GoogleClassShare\yelp_academic_dataset_review_preprosessd.json'

for i in [1000]:#,500,1000,2000]:       #runing the code for various samples sizes to calculate the metrics for each sample size
    samples = i
    start = timer()
    print("The number of samples under process  :%d"%samples)
    bag_of_words = classifier(file_name)
    print ("Processing time for %d samples using 10 cross validation is %f"%(samples,timer() - start))
    print ("-"*70, "\n")

