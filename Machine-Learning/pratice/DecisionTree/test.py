import os
import subprocess
import pandas as pd


content_list = []

for content in os.listdir("."):  # "." means current directory
    content_list.append(content)

content_list.remove('data')
content_list.remove('test.py')
content_list.remove('output')
print(content_list)

temp = None
temp_dict = {}
for content in content_list:
    cmd = 'python '+content+' 11'
    try:
        temp_dict[content] = subprocess.check_output(cmd, shell=True)
    except:
        pass

for content in content_list:
    k = str(temp_dict[content])
    k = k[2:len(k)-1]
    k = k.split(sep='\\r\\n')
    k = k[:-1]
    # print(k)
    temp_dict[content] = {}
    i =0
    while i < len(k):
        temp_dict[content][k[i]] = k[i+1]
        i += 2
# print(temp_dict)
df = pd.DataFrame.from_dict(data=temp_dict, orient='index')
writer = pd.ExcelWriter('./output/new.xlsx')
df.to_excel(writer)
writer.save()

# print(df.head())

