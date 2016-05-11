"""
1.Removing Stopwords, using sentence level pruning with word tokenizer and performing stemming
        result-- success
2.Implemented POS tagging

"""

#suppressing warnings
import warnings
warnings.filterwarnings("ignore")
#importing features
from timeit import default_timer as timer
import json
import nltk
import numpy as np
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from numpy import average
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
from nltk import pos_tag
stop_words = set(stopwords.words("english"))
stemmer = ps()

pos_filter = ['CC','CD','DT','EX','FW','IN','LS','MD','NN','NNS','NP','NPS','PDT','PP','PP$','SYM','TO','WDT','WP','WP$','WRB']
def pruneWords(review_text):
    review_sent_split = review_text.split('.')
    review_word_bag = []
    for sentence in review_sent_split:
        words = list(nltk.word_tokenize(sentence))
        pos_tagged_words = pos_tag(words)
        #print(pos_tagged_words)
        review_word_bag += [(stemmer.stem(tagged_word[0])) for tagged_word in pos_tagged_words if (tagged_word[0] not in stop_words and tagged_word[1] not in pos_filter)]
        #print(review_word_bag)
    return review_word_bag

#def vectAdd(vecta, vectb):
#    return vecta + vectb

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
        review_word_bag = pruneWords(review["text"])

        # updating bag of words when encountering a new word
        bag_of_words += [word for word in review_word_bag if word not in bag_of_words]

        review_sparse_vect.append(list(np.zeros(len(bag_of_words))))
        rating_sparse_vect.append(review['stars'])

        for word in review_word_bag:
            review_sparse_vect[i][bag_of_words.index(word)] = 1

        i=i+1
        if i >= samples:
            break
        line = f.readline()

    # increasing the size of vectors to bag of words
    for vect in review_sparse_vect:
        diff = len(bag_of_words) - len(vect)
        #vect = vectAdd(vect, np.zeros(len(diff)))
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
    error  = average(abs(predicted - rating_sparse_vect)/4)
    print("The fined grained Accuracy for this case is :   %d"%((1- error)*100))

file_name = 'D:\LargeDatasets\yelp\GoogleClassShare\yelp_academic_dataset_review_preprosessd.json'
for i in [100,500,1000,2000]:           #runing the code for various samples sizes to calculate the metrics for each sample size
    samples = i
    start = timer()
    print("The number of samples under process  :%d"%samples)
    bag_of_words = classifier(file_name)
    print ("Processing time for %d samples using 10 cross validation is %f"%(samples,timer() - start))
    print ("-"*70, "\n")