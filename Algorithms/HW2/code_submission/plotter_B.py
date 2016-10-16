# run the code in python 2.7

import time
import numpy as np
import sys
from math import log10
import matplotlib.pyplot as plt
from matplotlib import style
from random import randint
style.use('ggplot')
sys.setrecursionlimit(10000)

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

def randomized_pivot(p,r):
    return randint(p,r)

def randomized_partition(A, p, r):
    k = randomized_pivot(p, r)
    exchange(A[r], A[k])
    return partition(A, p, r)

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

def randomized_quick_sort(A, p, r):
    if p < r:
        q = randomized_partition(A, p, r)
        quick_sort(A, p, q-1)
        quick_sort(A, q+1, r)

def rquick_sort(unsorted_list):
    A = unsorted_list
    p = 0
    r = len(A) - 1
    randomized_partition(A, p, r)
    return A


def plot_graph(input_size_list, run_time_list):

    input_size_list = np.array(input_size_list)
    run_time_list = np.array(run_time_list)

    plt.semilogx(input_size_list, [0]*len(input_size_list))
    return plt.plot(input_size_list, run_time_list)


if __name__ == '__main__':
    k = int(sys.argv[1])
    i = 1
    input_size_list_quick_sort = list()
    run_time_list_quick_sort = list()
    input_size_list_rquick_sort = list()
    run_time_list_rquick_sort = list()

    while True:
        high = i
        low  = -high
        input_size = i

        unsorted_list  = generate_data(low, high, input_size)

        start = time.time()
        quick_sorted_list = quick_sort_algo(unsorted_list)
        stop = time.time()
        time_elapsed = stop - start
        # print (input_size, time_elapsed)
        input_size_list_quick_sort.append(input_size)
        run_time_list_quick_sort.append(time_elapsed)

        start = time.time()
        rquick_sorted_list = rquick_sort(unsorted_list)
        stop = time.time()
        time_elapsed = stop - start
        # print (input_size, time_elapsed)
        input_size_list_rquick_sort.append(input_size)
        run_time_list_rquick_sort.append(time_elapsed)

        if i > k:
            break
        elif len(input_size_list_quick_sort) < 2:
            i += 1
        else:
            i += input_size_list_quick_sort[-2]

    print (input_size_list_quick_sort, run_time_list_quick_sort)
    print (input_size_list_rquick_sort, run_time_list_rquick_sort)

    fig = plt.figure()
    plt.title('Quick Sort vs Randomized Quick Sort Running Time Comparisions')
    plt.grid(True)

    blue_line, = plot_graph(input_size_list_quick_sort, run_time_list_quick_sort)
    red_line, = plot_graph(input_size_list_rquick_sort, run_time_list_rquick_sort)

    plt.legend([blue_line, red_line], ['Quick sort', 'Randomized Quick sort'])

    plt.show()