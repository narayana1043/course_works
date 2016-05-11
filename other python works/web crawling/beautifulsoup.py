import requests
from bs4 import BeautifulSoup

try:
    #url = "https://accounts.yellowpages.com/login?next=https%3A%2F%2Faccounts.yellowpages.com%2Fdialog%2Foauth&client_id=590d26ff-34f1-447e-ace1-97d075dd7421&response_type=code&app_id=WEB&source=ypu_login&vrid=cc57ab37-b73a-490b-aa76-4b3775c46985&merge_history=true"
    url = "http://www.yellowpages.com/search?search_terms=coffee&geo_location_terms=Los+Angeles%2C+CA"
    #url_page2 = "http://www.yellowpages.com/search?search_terms=Coffee&geo_location_terms=Los%20Angeles%2C%20CA&page=2"
    r = requests.get(url, auth = ('narayana1043@gmail.com','Baba@123'))
    #print (r.status_code)
    #print(r.content)
    soup = BeautifulSoup(r.content)
    #print(soup.prettify())
    soup = BeautifulSoup(r.content)
    #print(soup.prettify())
    #print(soup.find_all("a"))
    #for link in soup.find_all("a"):
        #print(link.get("href"))
        #print(link.text , link.get("href"))
        #print ("<a href='%s'>%s</a>"%(link.get("href"), link.text))

    g_data = soup.find_all("div",{"class":"info"})
    print(g_data[0].contents[0].find_all("a",{"class":"business-name"})[0].text)
    for item in g_data:
        #print(item.text)
        print(item.contents[0].find_all("a", {"class":"business-name"})[0].text)
        #print(item.contents[1].find_all("p",{"class":"adr"})[0].text)
        try:
            print(item.contents[1].find_all("span",{"itemprop":"addressLocality"})[0].text.replace(",",""))
        except Exception:
            pass
        try:
            print(item.contents[1].find_all("span",{"itemprop":"addressRegion"})[0].text)
        except Exception:
            pass
        try:
            print(item.contents[1].find_all("span",{"itemprop":"postalCode"})[0].text)
        except Exception:
            pass
        try:
            print(item.contents[1].find_all("div",{"class":"primary"})[0].text)
        except Exception:
            pass
except Exception as e:
    print(str(e))

