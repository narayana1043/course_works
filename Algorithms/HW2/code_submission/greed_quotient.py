import sys
from random import randint

def exchange(x, y):
    return y, x

def randomized_pivot(p,r):
    return randint(p,r)

def partition(A, p, r):
    pivot = A[r]
    i = p-1
    for j in range(p, r):
        if A[j] >= pivot:
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


def rquick_sort_algo(unsorted_list, size_of_list):
    A = unsorted_list
    p = 0
    r = size_of_list - 1
    rquick_sort(A, p, r)
    return A

def check_gq(list, max_gq):
    while i > 0:
        if list[i-1] > max_gq:
            return False
        i = i-1
    return

def cal_gq(size_of_list, list):
    inverse_sorted_list = rquick_sort_algo(list, size_of_list)
    max_gq = size_of_list
    position = 0
    while True:
        while position < size_of_list:
            if inverse_sorted_list[position] < max_gq:
                break
            position = position + 1
        if position >= max_gq:
            return max_gq
        else:
            max_gq = max_gq - 1


if  __name__ == '__main__':
    size_of_list = int(sys.argv[1])
    list = sys.argv[2]
    list = list[1:]
    list = list[:-1]
    list = list.split(',')
    list = [int(list[index]) for index in range(len(list))]
    max_gq = cal_gq(size_of_list, list)
    print (max_gq)

