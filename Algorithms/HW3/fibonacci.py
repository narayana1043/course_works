def  fibonacci(n):
    if n < 3:
        return 1
    else:
        sum = fibonacci(n-1) + fibonacci(n-2)
    return sum

print fibonacci(3)