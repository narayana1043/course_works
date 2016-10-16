from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
from math import sqrt

import dataloader as dtl
import regressionalgorithms as algs
import plotfcns

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

if __name__ == '__main__':
    run = True

    #common constants and data loaders
    trainsize = 1000
    testsize = 5000
    numparams = 1
    numruns = 1
    trainset, testset = dtl.load_ctscan(trainsize, testsize)
    print('Loading Data...')

    while (run == True):
        print('1 for 2.a')
        print('2 for 2.b')
        print('3 for 2.c')
        print('4 for 2.d')
        print('5 for 2.e')
        print('6 for 2.f')
        print('7 for 2.g')
        print('8 for 2.h')
        print('0 for exit')
        qnum = input('Enter the question number to run:')

        #***********************************************************#

        if qnum == '1':
            # Question 2.a,

            regressionalgs = {'Random': algs.Regressor(),
                        'Mean': algs.MeanPredictor(),
                        'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                        'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                        'FSLinearRegression200': algs.FSLinearRegression({'features': range(200)}),
                        'FSLinearRegression300': algs.FSLinearRegression({'features': range(300)}),
                        'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)})
                     }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams,numruns))

            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}
            print()
            for learnername, learner in regressionalgs.items():
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error
                print()

        elif qnum == '2':
            # Question 2.b
            errorlist = list()


            regressionalgs = {
                'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                # 'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                # 'FSLinearRegression100': algs.FSLinearRegression({'features': range(100)}),
                # 'FSLinearRegression200': algs.FSLinearRegression({'features': range(200)}),
                # 'FSLinearRegression300': algs.FSLinearRegression({'features': range(300)}),
                # 'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)})
            }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams, numruns))

            trainset, testset = dtl.load_ctscan(trainsize, testsize)
            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}
            print()
            for learnername, learner in regressionalgs.items():

                print('Running learner = ' + learnername + ' on all parameters ')

                for numsplits in [1,5,10,20,50]:
                    print()
                    print('Number of Splits:', numsplits)
                    for split in range(numsplits):
                        # Reset learner, and give new parameters; currently no parameters to specify
                        learner.reset(params)
                        # Train model
                        learner.learn(trainset[0], trainset[1])
                        # Test model
                        predictions = learner.predict(testset[0])
                        error = geterror(testset[1], predictions)
                        errorlist.append(error)
                        # print('Error for ' + learnername + ': ' + str(error))
                        errors[learnername][p, r] = error

                    print('     The mean Error for ' + learnername + ': ' + str(np.mean(error)))
                    print('     The standard error for ' + learnername + ': ' + str(np.std(errorlist)/sqrt(len(errorlist))))
                print()

        elif qnum == '3' or qnum == '4':
            # Question 2.c, Question 2.d

            RegularizationParams = [0.01, 0.1, 1]
            for RegularizationParam in RegularizationParams:
                print('Running for Regularization Parameter ', RegularizationParam)
                regressionalgs = {
                            'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
                         }
                numalgs = len(regressionalgs)

                errors = {}
                for learnername in regressionalgs:
                    errors[learnername] = np.zeros((numparams,numruns))

                print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

                # Currently only using 1 parameter setting (the default) and 1 run
                p = 0
                r = 0
                params = {}
                for learnername, learner in regressionalgs.items():
                    # Reset learner, and give new parameters; currently no parameters to specify
                    learner.reset(params)
                    print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    learner.learn(trainset[0], trainset[1], RegularizationParam)
                    # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    print('Error for ' + learnername + ': ' + str(error))
                    errors[learnername][p,r] = error

        elif qnum == '5':
            # Question 2.e
            # trainsize = 1000
            # testsize = 3000
            # numparams = 1
            # numruns = 1

            regressionalgs = {
                        'MPLinearRegression385': algs.MPLinearRegression({'features': range(385)}),
                     }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams,numruns))

            # trainset, testset = dtl.load_ctscan(trainsize,testsize)
            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}

            for learnername, learner in regressionalgs.items():
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1], maxerror=0.01, maxfeatures=10)
                # Test model
                predictions = learner.predict(testset[0])

                error = geterror(testset[1], predictions)
                error = error
                print('Error for ' + learnername + ': ' + str(error/150))
                error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
                print(error1)
                errors[learnername][p,r] = error

        elif qnum == '6':
            # Question 2.f

            regressionalgs = {
                        'LassoLinearRegression385': algs.LASSOLinearRegression({'features': range(385)}),
                     }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams,numruns))

            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}
            print('Running LassoLinearRegression on all parameters')
            for metaparam in [0, 0.01, 0.1, 1]:
                print(' meta-parameter: ', metaparam)
                for learnername, learner in regressionalgs.items():
                    # Reset learner, and give new parameters; currently no parameters to specify
                    learner.reset(params)
                    # print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    learner.learn(trainset[0], trainset[1],metaparam=metaparam)
                    # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    print(' Error for ' + learnername  +' : ' + str(error))
                    error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
                    errors[learnername][p,r] = error

        elif qnum == '7':
            # Question 2.f
            trainsize = 3000
            testsize = 10000
            numparams = 1
            numruns = 1

            regressionalgs = {
                        'SGDLinearRegression385': algs.SGDLinearRegression({'features': range(385)}),
                     }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams,numruns))

            trainset, testset = dtl.load_ctscan(trainsize,testsize)
            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}
            for learnername, learner in regressionalgs.items():
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                epochs = [5,10,15,25,100,400,1000,2000]
                for epoch in epochs:
                    learner.learn(trainset[0], trainset[1],epochs=epoch)
                    # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    print('Error for ' + learnername  +'   '+ 'epoch['+str(epoch)+']' + ': ' + str(error))
                    error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
                    errors[learnername][p,r] = error

        elif qnum == '8':
            # Question 2.h
            trainsize = 3000
            testsize = 10000
            numparams = 1
            numruns = 1

            regressionalgs = {
                        'BGDLinearRegression385': algs.BGDLinearRegression({'features': range(385)}),
                     }
            numalgs = len(regressionalgs)

            errors = {}
            for learnername in regressionalgs:
                errors[learnername] = np.zeros((numparams,numruns))

            trainset, testset = dtl.load_ctscan(trainsize,testsize)
            print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))

            # Currently only using 1 parameter setting (the default) and 1 run
            p = 0
            r = 0
            params = {}
            for learnername, learner in regressionalgs.items():
                # Reset learner, and give new parameters; currently no parameters to specify
                learner.reset(params)
                print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                for tolerance in [1,2,3,4,5,6]:
                    # Train model
                    learner.learn(trainset[0], trainset[1], tolerance=10*math.exp(-tolerance))
                    # Test model
                    predictions = learner.predict(testset[0])

                    error = geterror(testset[1], predictions)
                    print('Tolerance of training error: 10*exp(-', tolerance,')')
                    print(' Error for ' + learnername + ': ' + str(error))
                    print(' Processed Data set(Times):  ',learner.getprocessedtimes())
                    error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
                    errors[learnername][p,r] = error

        elif qnum == '0':
            break

        else:
            print('Invalid Input.. Choose Again')

#**************************************************************************************************#

# if __name__ == '__main__':
#     # Question 2.a,
#     trainsize = 1000
#     testsize = 5000
#     numparams = 1
#     numruns = 1
#
#     regressionalgs = {#'Random': algs.Regressor(),
#                 #'Mean': algs.MeanPredictor(),
#                 # 'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
#                 # 'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
#                 'FSLinearRegression200': algs.FSLinearRegression({'features': range(200)}),
#                 # 'FSLinearRegression300': algs.FSLinearRegression({'features': range(300)}),
#                 # 'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)})
#              }
#     numalgs = len(regressionalgs)
#
#     errors = {}
#     for learnername in regressionalgs:
#         errors[learnername] = np.zeros((numparams,numruns))
#
#     trainset, testset = dtl.load_ctscan(trainsize,testsize)
#     print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#     # Currently only using 1 parameter setting (the default) and 1 run
#     p = 0
#     r = 0
#     params = {}
#     for learnername, learner in regressionalgs.items():
#     	# Reset learner, and give new parameters; currently no parameters to specify
#     	learner.reset(params)
#     	print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#     	# Train model
#     	learner.learn(trainset[0], trainset[1])
#     	# Test model
#     	predictions = learner.predict(testset[0])
#     	error = geterror(testset[1], predictions)
#     	print('Error for ' + learnername + ': ' + str(error))
#     	errors[learnername][p,r] = error


# if __name__ == '__main__':
#     # Question 2.b
#     errorlist = list()
#     for split in range(10):
#         trainsize = 1000
#         testsize = 5000
#         numparams = 1
#         numruns = 1
#
#         regressionalgs = {
#             # 'Random': algs.Regressor(),
#             # 'Mean': algs.MeanPredictor(),
#             # 'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
#             # 'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
#             # 'FSLinearRegression100': algs.FSLinearRegression({'features': range(100)}),
#             # 'FSLinearRegression200': algs.FSLinearRegression({'features': range(200)}),
#             'FSLinearRegression300': algs.FSLinearRegression({'features': range(300)}),
#             # 'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)})
#         }
#         numalgs = len(regressionalgs)
#
#         errors = {}
#         for learnername in regressionalgs:
#             errors[learnername] = np.zeros((numparams, numruns))
#
#         trainset, testset = dtl.load_ctscan(trainsize, testsize)
#         print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#         # Currently only using 1 parameter setting (the default) and 1 run
#         p = 0
#         r = 0
#         params = {}
#         for learnername, learner in regressionalgs.items():
#             # Reset learner, and give new parameters; currently no parameters to specify
#             learner.reset(params)
#             print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#             # Train model
#             learner.learn(trainset[0], trainset[1])
#             # Test model
#             predictions = learner.predict(testset[0])
#             error = geterror(testset[1], predictions)
#             errorlist.append(error)
#             print('Error for ' + learnername + ': ' + str(error))
#             errors[learnername][p, r] = error
#
#     print('The mean Error for ' + learnername + ': ' + str(np.mean(error)))
#     print('The standard error for ' + learnername + ': ' + str(np.std(errorlist)/sqrt(len(errorlist))))

#
# if __name__ == '__main__':
#     # Question 2.c, Question 2.d
#
#     RegularizationParams = [0.01, 0.1, 1]
#     for RegularizationParam in RegularizationParams:
#         print('Running for Regularization Parameter ', RegularizationParam)
#         trainsize = 1000
#         testsize = 5000
#         numparams = 1
#         numruns = 1
#
#         regressionalgs = {
#                     'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
#                  }
#         numalgs = len(regressionalgs)
#
#         errors = {}
#         for learnername in regressionalgs:
#             errors[learnername] = np.zeros((numparams,numruns))
#
#         trainset, testset = dtl.load_ctscan(trainsize,testsize)
#         print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#         # Currently only using 1 parameter setting (the default) and 1 run
#         p = 0
#         r = 0
#         params = {}
#         for learnername, learner in regressionalgs.items():
#             # Reset learner, and give new parameters; currently no parameters to specify
#             learner.reset(params)
#             print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#             # Train model
#             learner.learn(trainset[0], trainset[1], RegularizationParam)
#             # Test model
#             predictions = learner.predict(testset[0])
#             error = geterror(testset[1], predictions)
#             print('Error for ' + learnername + ': ' + str(error))
#             errors[learnername][p,r] = error


# if __name__ == '__main__':
#     # Question 2.e
#     trainsize = 5000
#     testsize = 5000
#     numparams = 1
#     numruns = 1
#
#     regressionalgs = {
#                 'MPLinearRegression385': algs.MPLinearRegression({'features': range(385)}),
#              }
#     numalgs = len(regressionalgs)
#
#     errors = {}
#     for learnername in regressionalgs:
#         errors[learnername] = np.zeros((numparams,numruns))
#
#     trainset, testset = dtl.load_ctscan(trainsize,testsize)
#     print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#     # Currently only using 1 parameter setting (the default) and 1 run
#     p = 0
#     r = 0
#     params = {}
#     for learnername, learner in regressionalgs.items():
#         # Reset learner, and give new parameters; currently no parameters to specify
#         learner.reset(params)
#         print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#         # Train model
#         learner.learn(trainset[0], trainset[1], maxerror=0.01, maxfeatures=10)
#         # Test model
#         predictions = learner.predict(testset[0])
#
#         error = geterror(testset[1], predictions)
#         print('Error for ' + learnername + ': ' + str(error))
#         error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
#         print(error1)
#         errors[learnername][p,r] = error

# if __name__ == '__main__':
#     # Question 2.g
#     trainsize = 1000
#     testsize = 5000
#     numparams = 1
#     numruns = 1
#
#     regressionalgs = {
#                 'SGDLinearRegression385': algs.SGDLinearRegression({'features': range(385)}),
#              }
#     numalgs = len(regressionalgs)
#
#     errors = {}
#     for learnername in regressionalgs:
#         errors[learnername] = np.zeros((numparams,numruns))
#
#     trainset, testset = dtl.load_ctscan(trainsize,testsize)
#     print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#     # Currently only using 1 parameter setting (the default) and 1 run
#     p = 0
#     r = 0
#     params = {}
#     for learnername, learner in regressionalgs.items():
#         # Reset learner, and give new parameters; currently no parameters to specify
#         learner.reset(params)
#         print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#         # Train model
#         epochs = [5,10,15,25]
#         for epoch in epochs:
#             learner.learn(trainset[0], trainset[1],epochs=epoch)
#             # Test model
#             predictions = learner.predict(testset[0])
#             error = geterror(testset[1], predictions)
#             print('Error for ' + learnername  +'   '+ 'epoch['+str(epoch)+']' + ': ' + str(error))
#             error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
#             errors[learnername][p,r] = error



# if __name__ == '__main__':
#     # Question 2.f
#     trainsize = 1000
#     testsize = 5000
#     numparams = 1
#     numruns = 1
#
#     regressionalgs = {
#                 'LassoLinearRegression385': algs.LASSOLinearRegression({'features': range(385)}),
#              }
#     numalgs = len(regressionalgs)
#
#     errors = {}
#     for learnername in regressionalgs:
#         errors[learnername] = np.zeros((numparams,numruns))
#
#     trainset, testset = dtl.load_ctscan(trainsize,testsize)
#     print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#     # Currently only using 1 parameter setting (the default) and 1 run
#     p = 0
#     r = 0
#     params = {}
#     for learnername, learner in regressionalgs.items():
#         # Reset learner, and give new parameters; currently no parameters to specify
#         learner.reset(params)
#         print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#         # Train model
#         learner.learn(trainset[0], trainset[1],metaparam=0.1)
#         # Test model
#         predictions = learner.predict(testset[0])
#         error = geterror(testset[1], predictions)
#         print('Error for ' + learnername  +' : ' + str(error))
#         error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
#         errors[learnername][p,r] = error

# if __name__ == '__main__':
#     # Question 2.h
#     trainsize = 1000
#     testsize = 5000
#     numparams = 1
#     numruns = 1
#
#     regressionalgs = {
#                 'BGDLinearRegression385': algs.BGDLinearRegression({'features': range(385)}),
#              }
#     numalgs = len(regressionalgs)
#
#     errors = {}
#     for learnername in regressionalgs:
#         errors[learnername] = np.zeros((numparams,numruns))
#
#     trainset, testset = dtl.load_ctscan(trainsize,testsize)
#     print(('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0]))
#
#     # Currently only using 1 parameter setting (the default) and 1 run
#     p = 0
#     r = 0
#     params = {}
#     for learnername, learner in regressionalgs.items():
#         # Reset learner, and give new parameters; currently no parameters to specify
#         learner.reset(params)
#         print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
#         # Train model
#         learner.learn(trainset[0], trainset[1])
#         # Test model
#         predictions = learner.predict(testset[0])
#
#         error = geterror(testset[1], predictions)
#         print('Error for ' + learnername + ': ' + str(error))
#         error1 = np.linalg.norm(np.subtract(predictions, testset[1]))/predictions.shape[0]
#         errors[learnername][p,r] = error


