from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
from scipy.stats.stats import pearsonr as pr
from math import sqrt


xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    # hm - how many datapoints
    # variance - how variable the data set will be
    # step - how far on average to step up 'y' value per point
    # correlation - positive(True)/negative(False)/none

    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_intercept(xs, ys):
    # reference: http://mathworld.wolfram.com/LeastSquaresFitting.html
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) ** 2) - mean(xs ** 2)))
    temp = pr(xs, ys)[0] * np.std(ys)/np.std(xs)  # correlation* (std of y / std of x)
    b = mean(ys) - m * mean(xs)

    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = mean(ys_orig)
    squared_error_regr = squared_error(ys_orig, ys_line)  # variance of error
    squared_error_y_mean = squared_error(ys_orig, y_mean_line) # variance of y
    return 1 - (squared_error_regr / squared_error_y_mean)


# xs, ys = create_dataset(hm=40,variance=80,step=2,correlation='pos')
# xs, ys = create_dataset(hm=40, variance=40, step=2, correlation='pos')
# xs, ys = create_dataset(hm=40,variance=10,step=2,correlation='pos')
# xs, ys = create_dataset(hm=40,variance=80,step=2,correlation=False)

m, b = best_fit_slope_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]

predict_x = 8
predict_y = (m * predict_x) + b
r_squared = coefficient_of_determination(ys, regression_line)
print("R squared",r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g', s=100)
plt.plot(xs, regression_line)
plt.show()
