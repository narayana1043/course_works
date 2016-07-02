import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8]
y = [2,4,5,6,7,8,6,5]

plt.scatter(x,y, label='scatter',color='k', marker='x', s=60)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.legend()
plt.show()