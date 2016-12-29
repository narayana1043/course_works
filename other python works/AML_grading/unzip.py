path = "E:\submissions\\"


import zipfile
import os
import rarfile
import gzip
import subprocess

files_list = os.listdir(path)

for file in files_list:

    if file[-3:] == 'zip':
        zip_ref = zipfile.ZipFile(path+file, 'r')
        os.makedirs(path+'unzipped\\'+file[:-3], 777, exist_ok=True)
        try:
            zip_ref.extractall(path+'unzipped\\'+file[:-3])
            zip_ref.close()
        except:
            print(file)

    elif file[-3:] == 'rar':
        rar_ref = rarfile.RarFile(path+file)
        os.makedirs(path+'unzipped\\'+file[:-3], 777, exist_ok=True)
        try:
            rar_ref.extractall(path+'unzipped\\'+file[:-3])
            rar_ref.close()
        except:
            # print(file)
            pass

    elif file[-2:] == 'gz':
        gz_ref = gzip.GzipFile(path+file)
        os.makedirs(path+'unzipped\\'+file[:-3], 777, exist_ok=True)
        try:
            gz_ref.extractall(path+'unzipped\\'+file[:-3])
            gz_ref.close()
        except:
            print(file)
            pass
    elif file[-2:] == '7z':
        os.makedirs(path + 'unzipped\\' + file[:-3], 777, exist_ok=True)
        print(file)
        # try:
        #     subprocess.call(r'"C:\Program Files\7-Zip\7z.exe" e '+ path+file + ' -o '
        #                     "path + 'unzipped\\' + file[:-3]")
        # except:
        #     print(file)
    else:
        print(file)
