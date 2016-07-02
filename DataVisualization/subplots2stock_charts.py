#  OHLC- open, high, low, close

import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ochl
from matplotlib import style
#style.use('ggplot')
style.use('fivethirtyeight')
# print(plt.style.available)
# style.use('dark_background')

# print(plt.__file__)       # location of matplot files


def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconvertor(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconvertor


def graph_data(stock):

    fig = plt.figure()
    ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=1, colspan=1)
    plt.title("EBAY")
    ax2 = plt.subplot2grid(shape=(6, 1), loc=(1, 0), rowspan=4, colspan=1)
    plt.xlabel('date')
    plt.ylabel('price')
    ax3 = plt.subplot2grid(shape=(6, 1), loc=(5, 0), rowspan=1, colspan=1)


    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=1m/csv'
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
                                                                              converters={0: bytespdate2num('%Y%m%d')})

    x = 0
    y = len(date)
    ohlc = []
    while x<y:
        append_me = date[x], open_price[x], high_price[x], low_price[x], close_price[x], volume[x]
        ohlc.append(append_me)
        x += 1

    candlestick_ochl(ax2, ohlc, width=0.4, colorup='g', colordown='r')  # can use hex colors

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax2.grid(True)

    bbox_props = dict(boxstyle='round', facecolor='y', edgecolor='k', lw=1)

    ax2.annotate(str(close_price[-1]), xy=(date[-1], close_price[-1]),
                 xytext=(date[-1]+4,close_price[-1]), bbox=bbox_props)


    # plt.title('Stock')
    # plt.legend()
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
    plt.show()


graph_data('EBAY')