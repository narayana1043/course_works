import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ochl
from matplotlib import style
style.use('fivethirtyeight')

MA1 = 10
MA2 = 30

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'vaild')
    return smas

def high_miinus_low(highs, lows):
    return highs-lows

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconvertor(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconvertor


def graph_data(stock):

    fig = plt.figure(facecolor='#f0f0f0')
    ax1 = plt.subplot2grid(shape=(6, 1), loc=(0, 0), rowspan=1, colspan=1)
    plt.title(stock)
    plt.ylabel('H-L')
    ax2 = plt.subplot2grid(shape=(6, 1), loc=(1, 0), rowspan=4, colspan=1, sharex=ax1)
    plt.ylabel('price')
    ax2v = ax2.twinx()
    ax3 = plt.subplot2grid(shape=(6, 1), loc=(5, 0), rowspan=1, colspan=1, sharex=ax1)
    plt.ylabel('MAvgs')


    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=1y/csv'
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

    ma1 = moving_average(close_price, MA1)
    ma2 = moving_average(close_price, MA2)
    start = len(date[MA2-1:])

    h_l = list(map(high_miinus_low, high_price, low_price))

    ax1.plot_date(date[-start:], h_l[-start:], '-', label='H-L')
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='lower'))

    candlestick_ochl(ax2, ohlc[-start:], width=0.4, colorup='g', colordown='r')  # can use hex colors

    ax2.grid(True)

    bbox_props = dict(boxstyle='round', facecolor='y', edgecolor='k', lw=1)

    ax2.annotate(str(close_price[-1]), xy=(date[-1], close_price[-1]),
                 xytext=(date[-1]+4,close_price[-1]), bbox=bbox_props)
    ax2v.plot([],[], color='#007983', label='Volume', alpha=0.4)
    ax2v.fill_between(date[-start:], 0, volume[-start:], facecolor='#007983', alpha=0.4)
    ax2v.axes.yaxis.set_ticklabels([])
    ax2v.grid(False)
    ax2v.set_ylim(0, 2*volume.max())

    ax2.yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))

    ax3.plot(date[-start:], ma1[-start:], linewidth=1, label=str(MA1)+'MA')
    ax3.plot(date[-start:], ma2[-start:], linewidth=1, label=str(MA2)+'MA')
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(prune='upper',nbins=5))

    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))

    ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:],
                     where=(ma1[-start:] < ma2[-start:]),
                     facecolor='r',edgecolor='r',alpha=0.5)
    ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:],
                     where=(ma1[-start:] > ma2[-start:]),
                     facecolor='g', edgecolor='g', alpha=0.5)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

    ax1.legend()
    leg = ax1.legend(loc=9, ncol=1, prop={'size':11})
    leg.get_frame().set_aplha = 0.4
    ax2v.legend()
    leg = ax2v.legend(loc=9, ncol=1, prop={'size': 11})
    leg.get_frame().set_aplha = 0.4
    ax3.legend()
    leg = ax3.legend(loc=8, ncol=2, prop={'size': 11})
    leg.get_frame().set_aplha = 0.4

    plt.show()
    fig.savefig('google.png', facecolor='#f0f0f0')


graph_data('GOOG')