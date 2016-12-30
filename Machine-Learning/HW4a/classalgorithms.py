from __future__ import division

import random

import numpy as np
import sys

import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning
        self.params['featuredistparams']={}
        self.params['featurescount'] = 0
        self.params['classlabelsprob'] = 0
        
    # TODO: implement learn and predict functions                  
    def learn(self, Xtrain, ytrain):
        if self.params['usecolumnones'] == True:
            featurecount = Xtrain.shape[1]
        elif self.params['usecolumnones'] == False:
            featurecount = Xtrain.shape[1]-1
        classlabelsprob = {}
        for Class in np.unique(ytrain):
            classlabelsprob[Class] = ytrain.tolist().count(Class)/ytrain.shape[0]
        featuredistparams = {}
        for Class in classlabelsprob:
            featuredistparams[Class]={}
            classmatchindexes = [index for index in range(len(ytrain.tolist())) if ytrain[index] == Class]
            for featureindex in range(featurecount):
                mu, sigma = utils.learndistribution(Xtrain[classmatchindexes,featureindex])
                featuredistparams[Class][featureindex] = {'mu':mu, 'sigma':sigma}
        parameters = {'featuredistparams':featuredistparams, 'featurescount':featurecount, 'classlabelsprob':classlabelsprob}
        self.resetparams(parameters)
        # print(self.params)

    def likelihoodcal(self, dpt, distparams, featurelength):
        if featurelength >= 0:
            mu = distparams[featurelength]['mu']
            sigma = distparams[featurelength]['sigma']
            j = utils.calculateprob(dpt[featurelength], mu, sigma)
            return j * self.likelihoodcal(dpt, distparams, featurelength-1)
        else:
            return 1

    def predict(self, Xtest):
        yhat = []
        for testdpt in Xtest:
            allclassprobs = {}
            classlabels = self.params['classlabelsprob']
            for Class in classlabels.keys():
                featurelength = self.params['featurescount']
                classprob = self.params['classlabelsprob'][Class]
                featuredistparams = self.params['featuredistparams'][Class]
                allclassprobs[Class] = self.likelihoodcal(testdpt, featuredistparams, featurelength-1) * classprob
            classproblist = [(value, key) for key, value in allclassprobs.items()]
            yhat.append(max(classproblist)[1])
        return np.array(yhat)

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        elif self.params['regularizer'] is 'elastic':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    def learn(self,Xtrain, ytrain):
        Xshape = Xtrain.shape
        regwgt = self.params['regwgt']
        w = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        err = float('INF')
        p = utils.sigmoid(np.dot(Xtrain, w))
        tolerance = 0.1
        numsamples = Xshape[0]
        XX_n = np.dot(Xtrain.T,Xtrain)/numsamples
        eeta = 1 / np.dot(2, np.linalg.norm((XX_n)))
        stepsize = 0.1
        while True:
            P = np.diag(p)
            I = np.identity(p.shape[0])

            # weight update based on type of regularizer selected
            if self.params['regularizer'] is 'l1':
                gradient = np.dot(Xtrain.T, np.subtract(ytrain, p))
                hessian_inv = -np.linalg.inv(np.dot(np.dot (Xtrain.T, np.dot(P, (I - P))), Xtrain))
                w = np.subtract(w, np.dot(hessian_inv, gradient))
                w = utils.proximalOperator(w, regwgt, eeta)

            elif self.params['regularizer'] is 'l2':
                gradient = np.dot(Xtrain.T, np.subtract(ytrain, p))+ regwgt*self.regularizer[1](w)
                hessian_inv = -np.linalg.inv(np.dot(np.dot (Xtrain.T, np.dot(P, (I - P))), Xtrain) + regwgt)
                w = np.subtract(w, stepsize*np.dot(hessian_inv, gradient))

            elif self.params['regularizer'] is 'elastic':
                gradient = np.dot(Xtrain.T, np.subtract(ytrain, p)) + regwgt * self.regularizer[1](w)
                hessian_inv = -np.linalg.inv(np.dot(np.dot(Xtrain.T, np.dot(P, (I - P))), Xtrain) + regwgt)
                w = np.subtract(w, stepsize * np.dot(hessian_inv, gradient))
                w = utils.proximalOperator(w, regwgt, eeta)

            else:
                gradient = np.dot(Xtrain.T, np.subtract(ytrain, p))
                hessian_inv = -np.linalg.inv(np.dot(np.dot(Xtrain.T,np.dot(P,(I - P))), Xtrain))
                w = np.subtract(w, np.dot(hessian_inv, gradient))
            p = utils.sigmoid(np.dot(Xtrain, w))

            newerr = np.linalg.norm(np.subtract(ytrain,p))
            if abs(err - newerr)<tolerance:
                break
            elif newerr - err > 0:
                stepsize /= 10
            err = newerr

        self.weights = w

        return self.weights

    def predict(self, Xtest, weights=None):
        if weights is None:
            weights = self.weights
        p = utils.sigmoid(np.dot(Xtest, weights))
        p = utils.threshold_probs(p)
        return p

class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                       'transfer': 'sigmoid',
                       'stepsize': 0.01,
                       'epochs': 10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.wi = None
        self.wo = None

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        self.ni = Xtrain.shape[1]
        self.nh = self.params['nh']
        self.no = 1                     # hard coding number of output layers
        self.nwst = (self.ni,self.nh,self.no)
        self.lnwst = len(self.nwst)
        self.w = []
        self.stepsize = 0.1#self.params['stepsize']
        for i, j in zip(self.nwst[1:], self.nwst[:-1]):
            self.w.append(np.random.normal(scale=0.1,size=(j,i)))
        self.wi = self.w[0]
        self.wo = self.w[1]
        numsamples = Xtrain.shape[0]
        Z = np.insert(Xtrain, Xtrain.shape[1], ytrain, axis=1)
        for epoch in range(self.params['epochs']):
            stepsizereset = self.params['stepsize']
            np.random.shuffle(Z)
            Y = Z[:, -1]
            X = np.delete(Z, -1, axis=1)
            for sample_index in range(numsamples):
                stepsize = stepsizereset / (sample_index + 1)
                self._updateW(X[sample_index, :], Y[sample_index], stepsize)

    def _updateW(self, x, y,stepsize):
        delta = []
        self._evaluate(inputs=x)
        for l in reversed(range(self.lnwst-1)):
            if l == self.lnwst - 2:
                next_delta = self.lo[l] - y.T
                delta.append(next_delta*self.dtransfer(self.li[l]))
            else:
                prev_delta = np.dot(self.w[l+1],delta[-1])
                delta.append(prev_delta*self.dtransfer(self.li[l]))

        for i in range(self.lnwst-1):
            if i == 0:
                lo = x.T
            else:
                lo = self.lo[i-1]
            l = np.atleast_2d(lo)
            deltas = np.atleast_2d(delta[-i+1])
            self.w[i] -= self.stepsize*np.dot(l.T,deltas)

    def predict(self, Xtest):
        ytest = []
        n = len(Xtest)
        ret = np.ones((n, 1))
        for p in range(Xtest.shape[0]):
            ret[p, :] = self._evaluate(Xtest[p, :])
            threshold = ret[p, :]
            if threshold >= 0.5:
                ytest.append(1)
            else:
                ytest.append(0)
        # print ret
        return ytest

    def _evaluate(self, inputs):
        """
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        self.li = []    #layer input
        self.lo = []    #layer ouput

        for i in range(self.lnwst-1):
            if i==0:
                # layer inputs or net input to the layer activation
                linp = np.dot(inputs, self.w[0])
            else:
                linp = np.dot(self.lo[-1], self.w[i])
            self.li.append(linp)
            self.lo.append(self.transfer(self.li[-1]))
        return self.lo[-1].T

class LogitRegAlternative(Classifier):
    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.0}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        Xshape = Xtrain.shape
        numsamples = Xshape[0]
        self.weights = np.zeros(Xshape[1])
        w = np.random.rand(Xshape[1])
        # w = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)
        olderr = float('INF')
        # p = (1/2)(1 + w^Tx/sqrt(1+w^Tx))
        xw = np.dot(Xtrain,w)
        sqrt_one_plus_xwSquare = utils.sqrt_one_plus_xwSquare
        one_plus_xwSquare = utils.one_plus_xwSquare
        p = 0.5*(1 + np.divide(xw,sqrt_one_plus_xwSquare(xw)))
        # olderr = np.linalg.norm(np.subtract(ytrain, p))
        stepsize = 1
        while stepsize>0.01:
            oldw = w
            xw = np.dot(Xtrain,w)
            w = w - stepsize*\
                    np.dot(Xtrain.T, (
                        (1-2*ytrain)/sqrt_one_plus_xwSquare(xw)
                        + (xw/one_plus_xwSquare(xw))))/numsamples
            p = 0.5 * (1 + np.divide(xw, sqrt_one_plus_xwSquare(xw)))
            err = np.linalg.norm(np.subtract(ytrain,p))
            if abs(err - olderr)<0.01:
                stepsize = stepsize/10
            else:
                olderr = err
        self.weights = oldw


    def predict(self, Xtest):
        xw = np.dot(Xtest, self.weights)
        sqrt_one_plus_xwSquare = utils.sqrt_one_plus_xwSquare
        p = 0.5 * (1 + xw/sqrt_one_plus_xwSquare(xw))
        p = utils.threshold_probs(p)
        return p

class Kmeans(Classifier):

    def __init__(self, parameters={}):
        # Default: no of centers = 10
        self.params = {'nc':10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)

    def learn(self, Xtrain, ytrain):

        ns = Xtrain.shape[0]                      # number of sample
        nc = self.params['nc']                     # number of centers
        pss = Xtrain[1].shape[0]                  # per sample shape
        c = utils.getRandCenters(Xtrain, nc, beta=np.ones(nc))[0]
        clusters = {}
        self.clusters, self.centroids, self.labels = [], [], []

        while True:

            # define clusters
            for cluster_id in range(nc):
                clusters[cluster_id] = []

            # assign samples to nearest clusters
            for sample_id in range(ns):
                dist_to_centriods = np.linalg.norm(np.tile(Xtrain[sample_id,], (nc,1))-c, axis=1)
                cluster_id = np.argmin(dist_to_centriods)
                clusters[cluster_id].append(sample_id)

            # recalibrate the weights
            old_c = c
            c = np.empty(c.shape)
            centroid = {}
            for cluster_id, cluster in clusters.items():
                if cluster == []:
                    centroid[cluster_id] = np.random.rand(pss)
                else:
                    centroid[cluster_id] = np.mean(Xtrain[cluster], axis=0)
                c[cluster_id] = np.array(centroid[cluster_id])

            # break the loop if oscillations are less than prefixed constant
            if np.sum(np.abs(c-old_c)) < 1:
                break

        for cluster_id, cluster in clusters.items():
            '''
            0 - ylabel
            1 - centriod
            '''
            if cluster == []:
                self.clusters.append(c[cluster_id])
                self.labels.append(ytrain[0])
                self.centroids.append(c[cluster_id])
                # self.cluster_labels_centroid.append([ytrain[0], c[cluster_id]])
            else:
                self.clusters.append(Xtrain[cluster])
                self.labels.append(utils.most_common(ytrain[cluster]))
                self.centroids.append(c[cluster_id])
                # self.cluster_labels_centroid.append([utils.most_common(ytrain[cluster]), c[cluster_id]])

        return self.centroids, self.clusters

    def predict(self, Xtest):

        P = []
        labels, centroids = self.labels, self.centroids
        # for cl in self.cluster_labels:
        #     labels.append(cl[0])
        #     centroids.append(cl[1])

        for sample in Xtest:
            replicated_sample = np.tile(sample, (len(labels),1))
            dist_to_centeriods = np.linalg.norm(replicated_sample - centroids, axis=1)
            selected_centriod_index = np.argmin(dist_to_centeriods)
            class_label = labels[selected_centriod_index]
            P.append(class_label)

        return P

class RBFLogitReg(LogitReg):

    def __init__(self, parameters={}):
        # Default: no regularization, 10 random centers
        self.params = {'regwgt': 0.0, 'regularizer': 'None',
                       'centroid_selection_Algo':'Random',
                       'kernel': 'gaussian', 'p':10, 'beta':1}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

        # supported center selection methods
        if self.params['centroid_selection_Algo'] == 'Random':
            self.csAlgo = utils.getRandCenters
        elif self.params['centroid_selection_Algo'] == 'Kmeans':
            self.csAlgo = utils.getKmeanCenters
        else:
            raise Exception('RBFLogitReg -> cannot handle center selection')

        # supported kernel transformations
        if self.params['kernel'] == 'gaussian':
            self.kernel = utils.gaussianKernel
        else:
            raise Exception('RBFLogitReg -> cannot handle kernel transformation method')

    def learn(self, Xtrain, ytrain):

        p = self.params['p']                        # new dimensions
        beta = self.params['beta']                  # variance for the case of random centers
        Xtrain_unbiased = utils.rmBias(Xtrain)      # removing bias if it exists
        self.c, self.var = self.csAlgo(Xtrain_unbiased, ytrain, p, beta)
        # RBF Transfer of data
        Xtrain_Trans = self.kernel(Xtrain_unbiased,
                                         self.c, self.var, beta)

        Xtrain_Trans_with_bias = utils.addBias(Xtrain_Trans)

        # Logistic Regression on the transformed Data
        super().reset(parameters=self.params)
        self.weights = super().learn(Xtrain_Trans_with_bias, ytrain)

    def predict(self, Xtest):

        Xtest_unbiased = utils.rmBias(Xtest)  # removing bias if it exists

        # RBF Transfer of data
        Xtest_Trans = self.kernel(Xtest_unbiased,
                                         self.c, self.var, self.params['beta'])
        Xtest_Trans_with_bias = utils.addBias(Xtest_Trans)

        predictions = super().predict(Xtest_Trans_with_bias, self.weights)

        return predictions







