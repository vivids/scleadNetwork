'''
Created on Oct 11, 2018

@author: deeplearning
'''
import os
def writeInfo2File(info,path):
    path_dir = path.split('/')
    length = len(path_dir)
    base_dir = ''
    for i in range(length-1):
        base_dir = os.path.join(base_dir,path_dir[i])
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    with open(path,'w') as f:
        for key in info.keys():
            f.write(str(key)+':'+str(info[key]))
            f.write('\n')

def readInfoFromFile(path):
    infoList={}
    with open(path,'r') as f:
        for line in f.readlines():
            splits = line.split(':')
            infoList[splits[0]]=splits[1].rstrip('\n')
    return infoList
            