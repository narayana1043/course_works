import urllib.request
import urllib.parse
import re


try:
    thisurl = "https://www.facebook.com/"
    headers = {}
    headers['User-Agent'] =  'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'
    values = {'q':'basic', 'submit':'search'}
    data = urllib.parse.urlencode(values)
    data = data.encode('utf-8')
    req = urllib.request.Request(thisurl,headers=headers)#,data=data)
    resp_data = urllib.request.urlopen(req)
    resp_data = resp_data.read()
    #print(str(resp_data))
    #paragraphs = re.findall(r'<p>(.*?)</p>',str(resp_data))
    paragraphs = re.findall(r'<p>(.*?)</p>',str(resp_data))
    for each_sent in paragraphs:
        print(each_sent)
except Exception as e:
    print(str(e))

