from __future__ import division  # floating point division
import math
import itertools
import operator
import random
import classalgorithms as algs
import numpy as np

def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def calculateprob(x, mean, stdev):
    if stdev < 1e-3:
        if math.fabs(x-mean) < 1e-2:
            return 1.0
        else:
            return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    xvec[xvec < -100] = -100
    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
 
    return vecsig

def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)

def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)

def dl2(vec):
    """ Gradient of l2 norm on a vector """
    return vec

def l1(vec):
    """ l1 norm on a vector """
    return np.linalg.norm(vec, ord=1)

def dl1(vec):
    """ Subgradient of l1 norm on a vector """
    grad = np.sign(vec)
    grad[abs(vec) < 1e-4] = 0.0
    return grad

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes

def logsumexp(a):
    """
    Compute the log of the sum of exponentials of input elements.
    Modified scipys logsumpexp implemenation for this specific situation
    """

    awithzero = np.hstack((a, np.zeros((len(a),1))))
    maxvals = np.amax(awithzero, axis=1)
    aminusmax = np.exp((awithzero.transpose() - maxvals).transpose())

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(aminusmax, axis=1))

    out = np.add(out,maxvals)

    return out

def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]

def learndistribution(nparray):
    mu, sigma = np.mean(nparray), np.std(nparray, ddof=1)
    return mu, sigma

def sqrt_one_plus_xwSquare(xw):
    return np.sqrt(1 + np.square(xw))

def one_plus_xwSquare(xw):
    return (1 + np.square(xw))

def proximalOperator(W, metaparam, eeta):
    constant = metaparam*eeta
    for w in range(W.shape[0]):
        if W[w] > constant:
            W[w] = W[w] - constant
        elif W[w] < (-1*constant):
            W[w] = W[w] + constant
        elif abs(W[w]) <= constant:
            W[w] = 0

    return W

def rmBias(ip):
    var = np.var(ip, axis=0)
    biases = np.where(var == 0)
    unbiased = [i for i in range(ip.shape[1])]
    unbiased.remove(biases[0])
    return ip[:, unbiased]

def addBias(ip):
    ns = ip.shape[0]
    ones = np.atleast_2d(np.ones(ns)).T
    return np.append(ip, ones, axis=1)

def most_common(L):
    #
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key= operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

def getRandCenters(Xtrain, nc, beta):
    '''
    Generates random centers in the given dataset
    :param Xtrain: given data set
    :param nc: number of center to generate
    :param beta: bandwidth
    :return: random centers
    '''
    ns = Xtrain.shape[0]                                                  # number of sample
    rand_sample_indices = random.sample(range(0, ns), nc)
    rand_centers = Xtrain[rand_sample_indices]
    rand_variance = np.ones(nc)*beta
    return rand_centers, rand_variance

def getKmeanCenters(Xtrain, ytrain, nc, beta):
    '''
    Generates centers in the given dataset using the k-means algorithm
    :param Xtrain: given data set
    :param nc: number of center to generate
    :return: random centers
    '''
    learner = algs.Kmeans()
    params = {'nc':nc}
    learner.reset(parameters=params)
    centroids, clusters = learner.learn(Xtrain, ytrain)

    variances = []

    for cluster in clusters:
        cluster_var = np.var(cluster, axis=0)
        cluster_var[cluster_var < 0.0001] = 0.0001
        variances.append(cluster_var)

    return np.array(centroids), np.array(variances)

def gaussianKernel(ip, centers, var, beta):
    '''
    Transform the data set using gaussian kernal
    :param input: Data set
    :param centers: number of centeres
    :param beta: 1/2(standard deviation)
    :return: Transformed Data set
    '''
    trans = np.empty([ip.shape[0], centers.shape[0]])
    var = var*beta
    var_inv = np.ones(var.shape)/var

    for c, index in zip(centers, range(centers.shape[0])):

        # calculate the eculdiean distance
        a = np.sum(np.sqrt(np.square(ip-c)*var_inv[index]), axis=1)/max(np.sum(np.sqrt(np.square(ip-c)*var_inv[index]), axis=1))
        # a = np.linalg.norm((ip-c) * var_inv[index], axis=1)

        trans[:,index] = beta*np.exp(-a)

    return trans