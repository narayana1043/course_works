import requests
from bs4 import BeautifulSoup

url = "http://www.yellowpages.com/search?search_terms=coffee&geo_location_terms=Los+Angeles%2C+CA"

r = requests.get(url)
soup = BeautifulSoup(r.content)
#print(soup.prettify)

g_data = soup.find_all("div",{"class" : "info"})