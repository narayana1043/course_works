path = 'E:\\PA2\\'

import os
import shutil

folders = os.listdir(path)

def change_permission(path_param):
    for dirpath, dirnames, filenames in os.walk(path_param):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            os.chmod(path, 0o777)

for file in folders:
    file += '\\'
    if os.path.isdir(path+file):
        inner_folders = os.listdir(path+file)
        print(inner_folders)
        count = len(inner_folders)
        if count == 1:
            source_name = path+file+inner_folders[0]
            dest_name = path+inner_folders[0]
            change_permission(source_name)
            if os.path.isdir(dest_name):
                change_permission(dest_name)
                shutil.rmtree(dest_name)
            os.rename(src=source_name, dst=dest_name)
            os.chmod(path+file, 0o777)
            os.rmdir(path+file)