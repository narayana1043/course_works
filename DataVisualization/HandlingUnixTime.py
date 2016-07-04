import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates
import datetime as dt


def bytesupdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)

    def bytesconvertor(b):
        s = b.decode(encoding)
        return strconverter(s)

    return bytesconvertor


def graph_data(stock):
    fig = plt.figure()
    ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0))

    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/' + \
                      stock + '/chartdata;type=quote;range=10d/csv'
    source_code = urllib.request.urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')

    for line in split_source:
        if len(line.split(',')) == 6 and 'values' not in line:
            stock_data.append(line)

            date, close_price, high_price, low_price, open_price, volume = \
                np.loadtxt(stock_data, dtype=np.float, delimiter=',',
                           unpack=True)
            dateconv = np.vectorize(dt.datetime.fromtimestamp)
            date = dateconv(date)

    ax1.plot_date(date, close_price, label='Price', fmt='-')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True, color='g', linestyle='-', linewidth=2)

    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90,
                        wspace=0.2, hspace=0)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('Internet Data plots')
    plt.legend()
    plt.show()


graph_data('TSLA')
