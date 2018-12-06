'''
Created on Oct 11, 2018

@author: deeplearning
'''

def writeInfo2File(info,path):
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
            