def set_operation(circle, k):
    while len(k) > 0:
        temp = k.pop()
        circle.remove(temp)
    # print(circle)
    return circle

circle = list(i for i in range(1, 101))
i = 1
k = list()
print("circle:", circle)

while len(circle) > 1:
    # print("before:",i)

    if len(circle)-1 >= i:
        k.append(circle[i])
        i += 2
    elif len(circle)-1 < i:
        i = (i % len(circle))
        print("k     ",k)
        circle = set_operation(circle, k)
        print("circle",circle)

        k = list()

        # print("after: ",i)

print(circle)