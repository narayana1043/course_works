from __future__ import division  # floating point division
import numpy as np
import math
import utilities as utils
from scipy.stats import pearsonr

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest
        
class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params={} ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    

    def learn(self, Xtrain, ytrain, RegularizationParam = 0):
        """
        Learns using the traindata
        :param Xtrain: Training Data of features
        :param ytrain: Trainig results
        :param RegularizationParam: if 0 normal linear regression, else regression with ridge regularlization
        :return: None
        """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(
            np.linalg.inv(np.add(np.dot(Xless.T,Xless),np.dot(RegularizationParam,np.identity(np.shape(Xless)[1])))),
            Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class MPLinearRegression(Regressor):
    '''
    Linear Regression with Maximum Pursuit approach(Greedy Algorithm)
    '''

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(params)

    def learn(self, Xtrain, Ytrain, maxerror=10, maxfeatures=None, reg=0.1):
            '''
            Learns using training data
            Regularlization based on MP greedy approach. Feature with best correlation with the residual gets selected
            in each run until the residual fall below some constant
            :param Xtrain: Training Data of features
            :param Ytrain: Trainig results
            :param maxresidual: residual thershold allowed
            :return: None
            '''
            if maxfeatures is None:
                maxfeatures = Xtrain.shape[1]
            featurecounter = 1
            Ytrain = np.reshape(Ytrain, (np.shape(Ytrain)[0],1))
            F = np.empty([np.shape(Xtrain)[0],1])           # selected features set
            W = np.zeros([1,1])                             # Weight Vector
            selectfeatures = list()                         # selected features list
            while True:
                cr = list()
                for param in range(Xtrain.shape[1]):
                    if param not in selectfeatures:
                        residue = np.subtract(np.dot(F, W), Ytrain)
                        residue = np.reshape(residue,(np.shape(residue)[0],))
                        # print(residue.shape)
                        # print(Xtrain[:,param].shape)5
                        cr.append([np.absolute(np.corrcoef(Xtrain[:,param],residue)[1,0]),param])
                maxcrfeature = max(cr, key=lambda x: x[0])
                crthershold = maxcrfeature[0]
                if crthershold < 0.01:
                    break
                maxparamindex = maxcrfeature[1]
                selectfeatures.append(maxparamindex)
                F = np.insert(F, F.shape[1],Xtrain[:,maxparamindex],axis=1)
                if featurecounter == 1:
                    F = np.delete(F, 0, axis=1)
                featurecounter = featurecounter + 1
                W = np.dot(np.dot(np.linalg.inv(np.add(np.dot(F.T, F),np.dot(reg,np.identity(np.shape(F)[1])))),F.T),Ytrain)
                Ypredicted = np.dot(F, W)
                error = np.linalg.norm(np.subtract(Ypredicted, Ytrain))/Ytrain.shape[0]
                print(error, featurecounter)
                if error < maxerror or featurecounter == Xtrain.shape[1] or featurecounter-1 == maxfeatures:
                    self.params = {'features':selectfeatures}
                    self.weights = W
                    break

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class BGDLinearRegression(Regressor):
    '''
    Linear regression with LASSO(least-absolute-squares-shrinkage-operator)
    '''

    def __init__(self, params):
        self.weights = None
        self.params = {'features':[1,2,3,4,5]}
        self.reset(params)
        self.processed_times = 0

    def linesearch(self, W, gradient, olderr, X, y):
        stepsize = 1
        while True:
            newW = np.subtract(W, np.dot(stepsize, gradient))
            newerr = utils.geterr(X, y, newW)
            if newerr < olderr:
                break
            else:
                stepsize = stepsize/2
        return newW, newerr

    def learn(self, Xtrain, ytrain, tolerance=10*math.exp(-4)):
        """
        Learns using the traindata
        :param Xtrain: Training Data of features
        :param ytrain: Trainig results
        :param RegularizationParam: if 0 normal linear regression, else regression with ridge regularlization
        :return: None
        """
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        W = np.random.rand(Xless.shape[1])
        olderr = 100000000000000
        newerr = utils.geterr(Xless, ytrain, W)
        while abs(olderr - newerr) > tolerance:
            olderr = newerr
            gradient = np.dot(Xless.T ,np.subtract(np.dot(Xless, W),ytrain))/numsamples
            W, newerr = self.linesearch(W, gradient, olderr, Xless, ytrain)
            self.processed_times += 1
        self.weights = W

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

    def getprocessedtimes(self):
        return self.processed_times

class SGDLinearRegression(Regressor):
    '''
    Linear regression with LASSO(least-absolute-squares-shrinkage-operator)
    '''

    def __init__(self, params):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(params)

    def learn(self, Xtrain, ytrain, epochs):
        """
        Learns using the traindata
        :param Xtrain: Training Data of features
        :param ytrain: Trainig results
        :param RegularizationParam: if 0 normal linear regression, else regression with ridge regularlization
        :return: None
        """
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        W = np.random.rand(Xless.shape[1])
        Z = np.insert(Xless, Xless.shape[1], ytrain, axis=1)
        stepsize = 0.01
        for epoch in range(epochs):
            stepsizereset = stepsize
            np.random.shuffle(Z)
            y = Z[:, -1]
            X = np.delete(Z, -1, axis=1)
            for sample_index in range(numsamples):
                newstepsize = stepsizereset / (sample_index + 1)
                err = np.subtract(np.dot(X[sample_index, :].T, W), y[sample_index])
                gradient = np.dot(err, X[sample_index, :])
                W = np.subtract(W, np.dot(gradient, newstepsize))

        self.weights = W

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class LASSOLinearRegression(Regressor):
    """
    Linear Regression Using Least Absolute Shrinkage selection operator
    """

    def __init__(self, params={}):
        self.weights = None
        self.params = {'features': [1, 2, 3, 4, 5]}
        self.reset(params)

    def learn(self, Xtrain, ytrain, metaparam):
        """
        Learns using the traindata
        :param Xtrain: Training Data of features
        :param ytrain: Trainig results
        :param RegularizationParam: if 0 normal linear regression, else regression with ridge regularlization
        :return: None
        """
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:, self.params['features']]
        W = np.zeros(Xless.shape[1])
        olderr = 100000000000000
        tolerance = 10 * math.exp(-2)
        XX_n = np.dot(1 / numsamples, np.dot(Xless.T, Xless))
        Xy_n = np.dot(1 / numsamples, np.dot(Xless.T, ytrain))
        eeta = 1 / np.dot(2, np.linalg.norm((XX_n)))
        newerr = np.dot(np.subtract(np.dot(Xless, W),ytrain).T, np.subtract(np.dot(Xless, W),ytrain))
        while abs(olderr - newerr) > tolerance:
            olderr = newerr
            newW = np.add(np.subtract(W, np.dot(eeta, np.dot(XX_n, W))),np.dot(eeta, Xy_n))
            W = utils.proximalOperator(newW, metaparam, eeta)
            newerr = np.dot(np.subtract(np.dot(Xless, W), ytrain).T, np.subtract(np.dot(Xless, W), ytrain))

        self.weights = W

    def predict(self, Xtest):
        Xless = Xtest[:, self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest


