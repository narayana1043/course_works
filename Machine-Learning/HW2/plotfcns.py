import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def plot_histogram(yvalues):
    plt.hist(yvalues, bins=50, facecolor='green')
    plt.xlabel('Regression variable')
    plt.ylabel('Probability')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()
