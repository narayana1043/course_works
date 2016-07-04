import mpl_toolkits
from mpl_toolkits.basemap import Basemap
from PIL import Image
import matplotlib.pyplot as plt

# m = Basemap(projection='mill',
#             llcrnrlat=-90,
#             llcrnrlon=-180,
#             urcrnrlat=90,
#             urcrnrlon=180,
#             resolution='l')   #resolution -- c,l,h,f

m = Basemap('mill', resolution='f')

# print(mpl_toolkits.basemap.__version__)
# print(mpl_toolkits.basemap._kw_args)


m.drawcoastlines()
m.drawcountries(linewidth=2)
# m.drawstates(color='b')
# m.drawcounties(color='r')
# m.etopo()
# m.bluemarble()
# m.fillcontinents()

plt.title('Basemap')
plt.show()
