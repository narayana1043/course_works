import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates
import datetime as dt

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconvertor(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconvertor


def graph_data(stock):

    fig = plt.figure()
    ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0))

    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
    source_code = urllib.request.urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')

    for line in split_source:
        if len(line.split(',')) == 6 and 'values' not in line:
            stock_data.append(line)

    date, close_price, high_price, low_price, open_price, volume = np.loadtxt(stock_data,
                                                                              dtype=np.float,
                                                                              delimiter=',',
                                                                              unpack=True,
                                                                              # %Y = full year. 2015
                                                                              # %y = partial year
                                                                              # %m = number month
                                                                              # %d = number day
                                                                              # %H = hours
                                                                              # %M = minutes
                                                                              # %s = seconds
                                                                              converters={0: bytespdate2num('%Y%m%d')})


    ax1.plot_date(date, close_price, label='Price',fmt='-')
    ax1.plot_date([],[], linewidth=5, label='loss', color='r', alpha=0.5,fmt='-')
    ax1.plot_date([], [], linewidth=5, label='gain', color='g', alpha=0.5,fmt='-')
    ax1.axhline(close_price[0], color='k', linewidth=2)
    ax1.fill_between(date, close_price,11, where=(close_price > close_price[0]),facecolor='g',alpha=0.3)
    ax1.fill_between(date, close_price, 11, where=(close_price < close_price[0]), facecolor='r',alpha=0.3)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax1.grid(True, color='g', linestyle='-', linewidth=2)
    ax1.xaxis.label.set_color('c')
    ax1.yaxis.label.set_color('r')
    ax1.set_yticks([0, 10, 20, 30 ,40])

    ax1.spines['left'].set_color('c')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax1.spines['left'].set_linewidth(4)
    ax1.tick_params(axis='x', colors='#f06215')     # color code('#f06215') from color hex website

    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('Stock')
    plt.legend()
    plt.show()


graph_data('EBAY')