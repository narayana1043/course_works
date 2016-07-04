import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:
    # yi(xi.w+b) >= 1 is the constraint function, our goal is to find w and b.
    #  we need w to be accurate but b need not be too precise.
    # w is the slope of hyperplane or line and it max value for any slope is
    #  infinite. so for w we start with a high value and we keep
    #   decreasing with steps sizes that are initially bigger but gradually
    #     decrease in length
    # b is the bias value of the constraint function its minimum value
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b] }  key- magnitude of w; values- list of w and b
        opt_dict = {}
        # transfroms will be applied on w. vector w is a 1x2 matrix
        # ex: w=[5,3] or [5,-3] or [-5,3] or [-5,-3]; magnitude[w]=sqrt(34)
        # for magnitude it doesn't matter but it matters for the direction
        #  of the hyperplane
        # so we check for all versions of w as shown above
        # every time we try to minimize the vector w and maximize the b
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
        # *this part can be avoided with other efficient funcs that might do* #
        # this is to get a maximum and minimum ranges both fot graph
        # and
        # where we are going start w and go stepping
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1; we will know when to stop by seeing
        #  how close we come to 1
        # big steps first; figure out the minimum of all big steps
        # and then go on taking smaller steps (avoiding pointless small steps)
        step_sizes = [self.max_feature_value * 1,
                      self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense: check this out commenting
                      #  the below line
                      # self.max_feature_value * 0.001,
                      ]

        # extremely expensive; we can tweak here 1,2 etc.
        # b is the bias;
        b_range_multiple = 5
        # b dosenot need to take smaller steps, it is not required to as
        # precise as w
        b_mulitiple = 5
        # latest_optimum is the first element in vector w, here is where we are
        #  cutting of the major edge
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because convex
            optimized = False
            while not optimized:
                # code in this for loop can be threaded to increase the
                #  performance
                for b in np.arange(-1 * (
                    self.max_feature_value * b_range_multiple),
                        self.max_feature_value * b_range_multiple,
                        step * b_mulitiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1 is the constraint function
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # we can even add a break here later; if we
                                    #  find a false there is no point in
                                    #   checking any longer
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            # np.linalg.norm(w_t) finding the magnitude of the
                            #  vector
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # according to the else statement the w keeps decreasing.
                # There is no point in decreasing it below zero because we
                # have already checked those values using the
                # trasformations. Hence once we reached a value for w below
                #  zero we can say that it is step optimized.
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            # taking the smallest w from the opt_dict
            print(opt_dict)
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # to go back to a point on the convex curve to start with the next
            #  step size
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                if not yi * (np.dot(w_t, xi) + b) >= 1:
                    found_option = False
                    print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*',
                            c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in
          data_dict[i]] for i in self.data]

        # hyperplane = x.w+b
        # v = x.w+b
        # positive support vector = 1
        # negative support vector = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (
            self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = 1
        # positive support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 1
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}

svm = SupportVectorMachine(visualization=True)
svm.fit(data=data_dict)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [6, -5],
              [5, 8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
