'''
Created on Nov 24, 2018

@author: deeplearning
'''
import os
import glob
import cv2
import constants as ct
from writeAndReadFiles import writeInfo2File

if __name__ == '__main__':
    dirtyDataNum = [0]
    sub_dirs = [x[0] for x in os.walk(ct.INPUT_DATA_DIR)]
    is_root_dir = True
    shape_360_180_total = 0
    shape_180_360_total = 0
    shape_256_256_total = 0
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
            shape_360_180 = 0
            shape_180_360 = 0
            shape_256_256 = 0
            for imgDir in file_list:
                img = cv2.imread(imgDir,0)
                row,col = img.shape
                if row/col >4/3:
                    shape_360_180 +=1
                elif row/col <3/4:
                    shape_180_360 +=1
                else:
                    shape_256_256 +=1
            shape_360_180_total += shape_360_180
            shape_180_360_total += shape_180_360
            shape_256_256_total += shape_256_256
            print(sub_dir)
            print('shape_360_180 is %g'%shape_360_180)
            print('shape_180_360 is %g'%shape_180_360)
            print('shape_256_256 is %g'%shape_256_256)
    
    print('shape_360_180_total is %g'%shape_360_180_total)
    print('shape_180_360_total is %g'%shape_180_360_total)
    print('shape_256_256_total is %g'%shape_256_256_total)
    
    dataSet = [shape_360_180_total,shape_180_360_total,shape_256_256_total]
    minNum = min(dataSet)
    dataSetProportion = [num//minNum for num in dataSet]
    dataSetProportionDict = {'shape_360_180':dataSetProportion[0],'shape_180_360':dataSetProportion[1],'shape_256_256':dataSetProportion[2]}
    writeInfo2File(dataSetProportionDict,ct.DATASET_INFO_DIR)
           
    
                    
       