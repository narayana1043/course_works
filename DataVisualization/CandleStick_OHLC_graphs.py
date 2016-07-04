#  OHLC- open, high, low, close

import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ochl


def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)

    def bytesconvertor(b):
        s = b.decode(encoding)
        return strconverter(s)

    return bytesconvertor


def graph_data(stock):
    fig = plt.figure()
    ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0))

    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + \
                      stock + '/chartdata;type=quote;range=3m/csv'
    source_code = urllib.request.urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')

    for line in split_source:
        if len(line.split(',')) == 6 and 'values' not in line:
            stock_data.append(line)

    date, close_price, high_price, low_price, open_price, volume = np.loadtxt(
        stock_data,
        dtype=np.float,
        delimiter=',',
        unpack=True,
        converters={0: bytespdate2num('%Y%m%d')})

    x = 0
    y = len(date)
    ohlc = []
    while x < y:
        append_me = date[x], open_price[x], high_price[x], low_price[x], \
                    close_price[x], volume[x]
        ohlc.append(append_me)
        x += 1

    candlestick_ochl(ax1, ohlc, width=0.4, colorup='g',
                     colordown='r')  # can use hex colors

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))

    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('Stock')
    plt.legend()
    plt.show()


graph_data('EBAY')
