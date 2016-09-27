import time
import numpy as np
import sys
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
    start = time.time()
    A = unsorted_list
    p = 0
    r = len(A) - 1
    quick_sort(A, p, r)
    stop = time.time()
    running_time = stop - start
    return A, running_time

if __name__ == '__main__':
    size = sys.argv[1]
    high = int(size)
    low = -high
    unsorted_list  = generate_data(low, high, int(size))
    print('unsorted_list    :', unsorted_list)
    sorted_list, time_elapsed = quick_sort_algo(unsorted_list)
    print('sorted_list      :',sorted_list)
    if time_elapsed == 0:
        print('time elapsed is less than 0.001 sec....')
        print('try length greater than 250')
    else:
        print('time_elapsed     :',time_elapsed)