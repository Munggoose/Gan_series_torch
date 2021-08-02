import os
import matplotlib.pyplot as plt
from PIL import Image


#재귀로 폴더내의 prefix 파일 불러오기
def reculsive_file_load(root_dir:str, prefix:str)->list:
    result = []
    files = os.listdir(root_dir)
    for file in files:
        path = os.path.join(root_dir, file)
        if file.split('.')[-1] == prefix:
            result.append(os.path.abspath(path))

        elif os.path.isdir(path):
            result += reculsive_file_load(path, prefix=prefix)
        
    return result
