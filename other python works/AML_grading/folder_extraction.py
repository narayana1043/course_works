path = 'E:\\submissions\\unzipped'
dest_folder = 'E:\\PA2'

import os
import shutil

folders = os.listdir(path)

def change_permission(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            os.chmod(path, 0o777)

for file in folders:
    rename = file.split('_')[-1]
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    try:
        if(path + '\\' + file+ '\\' +rename):
            source_name = temp_path = path + '\\' + file+ '\\' +rename
            dest_name = 'E:\\PA2\\'+rename
            if os.path.isdir(dest_name):
                change_permission(dest_name)
                shutil.rmtree(dest_name)
            change_permission(source_name)
            os.rename(src=source_name,dst=dest_name)
    except:
        source_name = temp_path = path + '\\' + file + '\\'
        dest_name = 'E:\\PA2\\' + rename
        if os.path.isdir(dest_name):
            shutil.rmtree(dest_name)
        os.chmod(source_name, mode=777)
        os.rename(src=source_name, dst=dest_name)
# print(folders)
