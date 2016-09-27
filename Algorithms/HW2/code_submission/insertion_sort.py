import time
import numpy as np
import sys

def generate_data(low, high, size=100):
    # generates integer data array
    data = np.random.random_integers(low, high, size)
    return data

def insertion_sort(unsorted_list):
    start = time.time()
    A = unsorted_list
    for j in range(2, A.size):
        key = A[j]
        # Insert A[j] into the sorted sequence A[1 .. j-1]
        i = j-1
        while i>0 and A[i]>key:
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    stop = time.time()
    running_time = stop - start
    return A, running_time

if __name__ == '__main__':
    size = sys.argv[1]
    high = int(size)
    low = -high
    unsorted_list  = generate_data(low, high, int(size))
    print('unsorted_list    :', unsorted_list)
    sorted_list, time_elapsed = insertion_sort(unsorted_list)
    print('sorted_list      :',sorted_list)
    if time_elapsed == 0:
        print('time elapsed is less than 0.001 sec....')
        print('try length greater than 250')
    else:
        print('time_elapsed     :',time_elapsed)
