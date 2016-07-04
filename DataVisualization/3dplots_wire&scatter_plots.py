from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [5, 6, 7, 8, 2, 5, 6, 3, 4, 6]
z = [3, 4, 5, 6, 7, 3, 5, 7, 8, 9]

ax1.plot_wireframe(x, y, z)

x2 = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
y2 = [-5, -6, -7, -8, -2, -5, -6, -3, -4, -6]
z2 = [3, 4, 5, 6, 7, 3, 5, 7, 8, 9]

ax1.scatter(x, y, z, c='g', marker='o', s=30)
ax1.scatter(x2, y2, z2, c='r', marker='*', s=30)
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()
