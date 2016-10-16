# run the code in python 2.7

import time
import numpy as np
import sys
from math import log10
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
sys.setrecursionlimit(10000)

def generate_data(size=100):
    # generates integer data array
    data = np.array([i for i in range(size)])
    return data

def insertion_sort(unsorted_list):
    A = unsorted_list
    for j in range(2, A.size):
        key = A[j]
        # Insert A[j] into the sorted sequence A[1 .. j-1]
        i = j-1
        while i>0 and A[i]>key:
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    return A

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

    input_size_list = np.array(input_size_list)
    run_time_list = np.array(run_time_list)

    plt.semilogx(input_size_list, [0]*len(input_size_list))
    return plt.plot(input_size_list, run_time_list)


if __name__ == '__main__':
    k = int(sys.argv[1])
    i = 1
    input_size_list_insertion_sort = list()
    run_time_list_insertion_sort = list()
    input_size_list_quick_sort = list()
    run_time_list_quick_sort = list()

    while True:
        input_size = i

        unsorted_list  = generate_data(input_size)
        start = time.time()
        insertion_sorted_list = insertion_sort(unsorted_list)
        stop = time.time()
        time_elapsed = stop - start
        # print (input_size, time_elapsed)
        input_size_list_insertion_sort.append(input_size)
        run_time_list_insertion_sort.append(time_elapsed)

        start = time.time()
        quick_sorted_list = quick_sort_algo(unsorted_list)
        stop = time.time()
        time_elapsed = stop - start
        # print (input_size, time_elapsed)
        input_size_list_quick_sort.append(input_size)
        run_time_list_quick_sort.append(time_elapsed)

        if i > k:
            break
        elif len(input_size_list_insertion_sort) < 2:
            i += 1
        else:
            i += input_size_list_quick_sort[-2]

    print (input_size_list_insertion_sort, run_time_list_insertion_sort)
    print (input_size_list_quick_sort, run_time_list_quick_sort)

    fig = plt.figure()
    plt.title('Insertion sort vs Quick Sort Running Time Comparisions for sorted input')
    plt.grid(True)

    blue_line, = plot_graph(input_size_list_insertion_sort, run_time_list_insertion_sort)
    red_line, = plot_graph(input_size_list_quick_sort, run_time_list_quick_sort)

    plt.legend([blue_line, red_line], ['Insertion sort', 'Quick sort'])

    plt.show()