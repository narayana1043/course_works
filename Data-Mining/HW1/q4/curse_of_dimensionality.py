import random
from statistics import mean
from math import log10
from scipy.spatial import distance
from matplotlib import pyplot as pp
from cycler import cycler as cyc

def cal_dmin_dmax(array,data_points):
    diff = []
    for row in range(data_points):
        for x in range(row+1,data_points):
            dist = distance.euclidean(array[row],array[x])
            diff.append(dist)
    return min(diff),max(diff)

def cal_r(dmin,dmax,r):
    if dmin != 0 and dmax > 0:
        return log10((dmax - dmin)/dmin)
    else:
        return r[-1]
def avg_r(r, multiple_run):
    temp = [r.pop(0) for i in range(multiple_run)]
    return(mean(temp))

def start_dimensionality(multiple_run, data_points):
    r_avg_array = []
    r = []
    if data_points < 10000:
        for column in range(1,101):
            for k in range(multiple_run):
                temp_array = [[random.random() for col in range(column)]for row in range(data_points)]
                dmin,dmax = cal_dmin_dmax(temp_array, data_points)
                #print(temp_array[0])
                r.append(cal_r(dmin, dmax, r))
            r_avg = avg_r(r, multiple_run)
            print(r_avg)
            r_avg_array.append(r_avg)
    else:
        for column in range(1,101):
            if column < 20:
                for k in range(multiple_run):
                    temp_array = [[random.random() for col in range(column)]for row in range(data_points)]
                    dmin,dmax = cal_dmin_dmax(temp_array, data_points)
                    #print(temp_array[0])
                    r.append(cal_r(dmin, dmax, r))
                r_avg = avg_r(r, multiple_run)
                print(r_avg)
                r_avg_array.append(r_avg)
            else:
                column += 10
                for k in range(multiple_run):
                    temp_array = [[random.random() for col in range(column)]for row in range(data_points)]
                    dmin,dmax = cal_dmin_dmax(temp_array, data_points)
                    #print(temp_array[0])
                    r.append(cal_r(dmin, dmax, r))
                r_avg = avg_r(r, multiple_run)
                print(r_avg)
                r_avg_array.append(r_avg)
    return r_avg_array

def start():
    x = 1
    y = 10
    pp.rc('axes', prop_cycle=(cyc('color', ['r','g','b'])))
    for i in range(1):
        r_avg_array = start_dimensionality(x,y)
        pp.plot(range(1,101), r_avg_array)
        y *= 10

    pp.ylabel('r(k) value')
    pp.xlabel('Dimentionality')
    pp.title('Data set - 1000')

    pp.axis([0, 100 , -2, 12 ])
    pp.show()

start()



