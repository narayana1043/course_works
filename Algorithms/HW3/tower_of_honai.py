Rod1 = [5]
Rod2 = []
Rod3 = []

def recursion(length, A, B, C):
    if length > 0:
        recursion(length -1, A, C, B)
        B.append(A.pop())
        print A,B,C
        recursion(length -1, C, B, A)
    return B


print recursion(len(Rod1), Rod1, Rod2, Rod3)