'''
Created on Nov 24, 2018

@author: deeplearning
'''
import os
import glob
import cv2
import shutil

INPUT_DATA_DIR ='/home/deeplearning/datasets/alarmClassification/experiment'
DIRTY_DATA_DIR = '/home/deeplearning/datasets/alarmClassification/dirtyData'

def movefile(srcfile,dstfile,dirtyDataNum):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,_=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile,dstfile)
        dirtyDataNum[0]+=1
        print("move %s -> %s"%( srcfile,dstfile))

if __name__ == '__main__':
    dirtyDataNum = [0]
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_DIR)]
    is_root_dir = True
    for sub_dir in sub_dirs :
            if is_root_dir :
                is_root_dir =False
                continue
            extensions = ['jpg','jpeg','JPG','JPEG']    
            file_list = []
    
            for extension in extensions:
                file_glob = os.path.join(sub_dir,'*.'+extension)
                file_list.extend(glob.glob(file_glob))
    
            if not file_list:
                continue
            for imgDir in file_list:
                img = cv2.imread(imgDir,0)
                width,height = img.shape
                if height/width >10 or width/height>10:
                    label_name = os.path.basename(sub_dir)
                    img_name = os.path.basename(imgDir)
                    movefile(imgDir, os.path.join(DIRTY_DATA_DIR,label_name,img_name),dirtyDataNum)
    print('%d dirty image is moved, the processing is finished!'%dirtyDataNum[0])
                    
       