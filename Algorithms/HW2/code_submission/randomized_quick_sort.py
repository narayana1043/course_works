import time
import numpy as np
import sys
from random import randint

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
    return partition(A, p, r)

def rquick_sort(A, p, r):
    if p < r:
        q = randomized_partition(A, p, r)
        rquick_sort(A, p, q-1)
        rquick_sort(A, q+1, r)


def rquick_sort_algo(unsorted_list):
    start = time.time()
    A = unsorted_list
    p = 0
    r = len(A) - 1
    rquick_sort(A, p, r)
    stop = time.time()
    running_time = stop - start
    return A, running_time

if __name__ == '__main__':
    size = sys.argv[1]
    high = int(size)
    low = -high
    unsorted_list  = generate_data(low, high, int(size))
    print('unsorted_list    :', unsorted_list)
    sorted_list, time_elapsed = rquick_sort_algo(unsorted_list)
    print('sorted_list      :',sorted_list)
    if time_elapsed == 0:
        print('time elapsed is less than 0.001 sec....')
        print('try length greater than 250')
    else:
        print('time_elapsed     :',time_elapsed)