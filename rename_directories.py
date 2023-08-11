import os
import re

path = 'New_Data_CoV2'
not_folders_renamed = 0
for folder in os.listdir(path):
    sub_path=path+"/"+folder
    
    for folder2 in os.listdir(sub_path):
       sub_path2 = sub_path + '/' + folder2
       count = 0
       for files in os.listdir(sub_path2):
            file = sub_path2 + '/' + files
            os.rename(file, sub_path2+ '/' + 'Anota' + str(count) + '.png')
            count += 1       
print("folders that haven't been renamed: ", not_folders_renamed)