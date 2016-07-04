import matplotlib.pyplot as plt

x = [2, 4, 6, 8, 10]
y = [6, 7, 8, 2, 4]

x2 = [1, 2, 5, 9, 11]
y2 = [7, 8, 4, 3, 2]

plt.bar(x, y, label='Bars1', color='g')
plt.bar(x2, y2, label='Bars2', color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Bar chart\nCheck it out')
plt.legend()
plt.show()
