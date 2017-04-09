# unix to date time stamp
"""
two important functions
strptime = converting string time to datetime that python understands
strftime = converting python datetime to string time that we understand
"""
from datetime import datetime as dt
import time

variable = dt.strptime('2010-01-01','%Y-%m-%d')
#print(datetime.datetime.fromtimestamp(variable).strftime('%Y-%m-%d %H:%M:%S'))
print(variable.strftime('%U'))


