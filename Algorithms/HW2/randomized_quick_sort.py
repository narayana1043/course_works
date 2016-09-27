import time
import numpy as np
import sys
from math import log10
from random import randint
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from matplotlib import style
style.use('fivethirtyeight')

def generate_data(low, high, size=100):
    # generates integer data array
    data = np.random.random_integers(low, high, size)
    return data

def exchange(x, y):
    return y, x

def randomized_pivot(p,r):
    return randint(p,r)

def partition(A, p, r):
    pivot = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] <= pivot:
            i = i + 1
            A[i], A[j] = exchange(A[i], A[j])
    A[i + 1], A[r] = exchange(A[i+1], A[r])
    return i+1

def randomized_partition(A, p, r):
    k = randomized_pivot(p, r)
    exchange(A[r], A[k])
    return randomized_partition(A, p, r)

def quick_sort(A, p, r):
    print(A)
    if p < r:
        q = partition(A, p, r)
        quick_sort(A, p, q-1)
        quick_sort(A, q+1, r)


def quick_sort_algo(unsorted_list):
    start = time.time()
    A = unsorted_list
    p = 0
    r = len(A) - 1
    quick_sort(A, p, r)
    stop = time.time()
    running_time = stop - start
    return A, running_time

def plot_graph(input_size_list, run_time_list):
    fig = plt.figure()
    plt.grid(True)

    input_size_list = np.array(input_size_list)
    input_size_list_new = np.linspace(input_size_list.min(), input_size_list.max(), 3000)

    run_time_list = np.array(run_time_list)
    run_time_smooth  = spline(input_size_list, run_time_list, input_size_list_new)

    plt.semilogx(input_size_list_new, run_time_smooth)

    # new_xticks = list()
    # for input_size in input_size_list:
    #     new_xticks.append(log10(input_size))
    # plt.xticks(input_size_list, new_xticks)

    plt.show()

if __name__ == '__main__':
    size = sys.argv[1]
    high = 10**int(size)
    low  = -high
    input_size_list = list()
    run_time_list = list()
    for input_size in range(1,int(size)+1):
        unsorted_list  = generate_data(low, high, 10**input_size)
        sorted_list, time_elapsed = quick_sort_algo(unsorted_list)
        print(unsorted_list)
        print(sorted_list)
        input_size_list.append(10**input_size)
        run_time_list.append(time_elapsed)
    # plot_graph(input_size_list, run_time_list)
