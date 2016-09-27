import time
import numpy as np
import sys
from math import log10
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.interpolate import interp1d
from matplotlib import style
style.use('ggplot')

def generate_data(low, high, size=100):
    # generates integer data array
    data = np.random.random_integers(low, high, size)
    return data

def exchange(x, y):
    return y, x

def partition(A, p, r):
    pivot = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= pivot:
            i = i + 1
            A[i], A[j] = exchange(A[i], A[j])
    A[i + 1], A[r] = exchange(A[i+1], A[r])
    return i+1

def quick_sort(A, p, r):
    if p < r:
        q = partition(A, p, r)
        quick_sort(A, p, q-1)
        quick_sort(A, q+1, r)


def quick_sort_algo(unsorted_list):
    A = unsorted_list
    p = 0
    r = len(A) - 1
    quick_sort(A, p, r)
    return A

def plot_graph(input_size_list, run_time_list):
    fig = plt.figure()
    plt.grid(True)
    input_size_list = np.array(input_size_list)
    run_time_list = np.array(run_time_list)

    plt.semilogx(input_size_list, [0]*len(input_size_list))
    plt.plot(input_size_list, run_time_list)
    plt.show()

if __name__ == '__main__':
    k = int(sys.argv[1])
    i = 10
    input_size_list = list()
    run_time_list = list()
    while (i < k):
        high = i
        low  = -high
        input_size = i
        # for input_size in range(1,int(size)+1):
        if i<80:
            pass
        else:
            unsorted_list  = generate_data(low, high, input_size)
            start = time.time()
            sorted_list = quick_sort_algo(unsorted_list)
            stop = time.time()
            time_elapsed = stop - start
            print(input_size, time_elapsed)
            # print(unsorted_list)
            # print(sorted_list)
            input_size_list.append(input_size)
            run_time_list.append(time_elapsed)
        if len(input_size_list) <= 2:
            i += i
        else:
            i += input_size_list[-2]
    print(input_size_list, run_time_list)
    plot_graph(input_size_list, run_time_list)