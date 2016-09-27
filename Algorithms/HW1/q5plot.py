e = 0.00001

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

list_n= list()
list_m = list()
n = 1
for m in range(0, 11):
    m /= 10
    list_m.append(m)
    while ((2**m) > (1 + e)**n):
        n += 1
    list_n.append(n)
    print(m, n)
for m in range(2, 1000, 30):
    list_m.append(m)
    while ((2**m) > (1 + e)**n):
        n += 1
    list_n.append(n)
    print(m, n)
m = np.array(list_m)
n = np.array(list_n)

n_smooth = np.linspace(n.min(), n.max(), 300)
m_smooth = spline(m, n, n_smooth)
plt.title("M versus n* graph", size=20 )
plt.xlabel("n*", size=25)
plt.xticks(size=15)
plt.yticks(size=15)
y = plt.ylabel("M", rotation=90, size=25)
y.set_rotation(90)
plt.plot(n, m)
plt.show()
