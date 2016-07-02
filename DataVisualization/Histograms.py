import matplotlib.pyplot as plt

population_ages = [22,55,23,12,34,123,113,34,212,45,12,43,1,13,45
                        ,21,1,12,14,15,14,12,111,123,65,77,88,99]

# bar plot
'''
ids = [x for x in range(len(population_ages))]

plt.bar(ids, population_ages)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
'''

# histogram

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(x=population_ages, bins=bins,histtype='bar',rwidth=0.5,label='His')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Histogram')
plt.legend()
plt.show()
