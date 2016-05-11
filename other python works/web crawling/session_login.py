import requests
from bs4 import BeautifulSoup

username = "narayana1043@gmail.com"
password = "Baba@123"
type = "submit"

try:
    #starting is to have presistent cookies
    URL = "https://accounts.yellowpages.com/login?next=https%3A%2F%2Faccounts.yellowpages.com%2Fdialog%2Foauth&client_id=590d26ff-34f1-447e-ace1-97d075dd7421&response_type=code&app_id=WEB&source=ypu_login&vrid=cc57ab37-b73a-490b-aa76-4b3775c46985&merge_history=true"
    session = requests.Session()
    #Authenticating
    #r = session.post(url, data={username,password,type})
    #r = requests.Session.post(url = url, auth = ('narayana1043@gmail.com','Baba@123'))
    r = session.post(URL,data = {username,password})
    print(r)
    soup = BeautifulSoup(r.content)
    print(soup.prettify())


except Exception as e:
    print(str(e))