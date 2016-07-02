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
    ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0))

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

    candlestick_ochl(ax1, ohlc, width=0.4, colorup='g', colordown='r')  # can use hex colors

    # ax1.plot(date, close_price)
    # ax1.plot(date, open_price)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.grid(True)

    # sample 1

    #
    # # annotation example with arrow
    # ax1.annotate('Sample Text!', xy=(date[11], high_price[11]),
    #              xytext=(0.8,0.9), textcoords='axes fraction',
    #              arrowprops= dict(facecolor='grey', color='grey'))
    # # warning due overriding the style settings of matplotlib style
    #
    # # font dict example
    # font_dict = {'family':'serif',
    #              'color':'darkred',
    #              'size': 15}
    #
    # # Hard coded text
    # ax1.text(date[10], close_price[1], 'Text Examples', fontdict=font_dict)

    # sample Two
    bbox_props = dict(boxstyle='round', facecolor='y', edgecolor='k', lw=1)
    ax1.annotate(str(close_price[-1]), xy=(date[-1], close_price[-1]),
                 xytext=(date[-1]+4,close_price[-1]), bbox=bbox_props)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('Stock')
    # plt.legend()
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
    plt.show()


graph_data('EBAY')