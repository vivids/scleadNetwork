'''
Created on Oct 7, 2018

@author: deeplearning
'''
import os
import glob
import tensorflow as tf
import numpy as np
import threading
import cv2

TEST_PERCENTAGE = 0
VALIDATION_PERCENTAGE = 10
INPUT_DATA_DIR ='/home/deeplearning/datasets/alarmClassification'
# INPUT_DATA_DIR ='/home/deeplearning/datasets/haha'
OUTPUT_TFRECORD_DIR = 'output/tfrecord'
CATELOGS = ('training','testing','validation')
INPUT_SIZE=1024
NUM_THREAD=4
# mutex = threading.Lock()

def create_image_lists(testing_percentage,validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA_DIR)]
#     print(sub_dirs)
    training_image = []
    testing_image = []
    validation_image = []
    is_root_dir= True
    for sub_dir in sub_dirs :
        if is_root_dir :
            is_root_dir =False
            continue
        extensions = ['jpg','jpeg','JPG','JPEG']    
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(sub_dir,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
#         print(file_list[1])   
        if not file_list:
            continue
        
        label_name = dir_name
        while file_list:
            file = file_list.pop()
            fileNameSplit = file.split('.')
            histOrCurr=fileNameSplit[0][-4:]
            if histOrCurr == 'curr':
                currFileName = file
                histFileName = fileNameSplit[0][:-4]+'hist.'+fileNameSplit[1]
                checkExist=histFileName
            else:
                histFileName = file
                currFileName = fileNameSplit[0][:-4]+'curr.'+fileNameSplit[1]
                checkExist = currFileName
            if checkExist in file_list:
                file_list.remove(checkExist)
                chance = np.random.randint(100)
                if chance < testing_percentage:
                    testing_image.append([currFileName,histFileName,label_name])
                elif chance < testing_percentage+validation_percentage:
                    validation_image.append([currFileName,histFileName,label_name])
                else:
                    training_image.append([currFileName,histFileName, label_name])   
            else:
                print(checkExist+' is lost, please check')
            
    result={
        CATELOGS[0]:training_image,
        CATELOGS[1]:testing_image,
        CATELOGS[2]:validation_image
        }
    return result    

def get_image_path(image_lists, image_dir,  label_name,index,category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path

def isFileExist(filePath):
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    else:
        files = os.listdir(filePath)
        for file in files:
            path = os.path.join(filePath,file)
            if os.path.isfile(path):
                os.remove(path)
 
    return filePath

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def convert_image_examples(rootDir, currImage, histImage,label): 
#     curr_img_raw=tf.gfile.FastGFile(os.path.join(rootDir,currImage),'rb').read()
#     hist_img_raw = tf.gfile.FastGFile(os.path.join(rootDir,histImage),'rb').read()    
#     curr_img=tf.image.convert_image_dtype(tf.image.decode_jpeg(curr_img_raw, channels=1),dtype=tf.float32)
#     hist_img=tf.image.convert_image_dtype(tf.image.decode_jpeg(hist_img_raw, channels=1),dtype=tf.float32)    
#     curr_img=tf.image.resize_images(curr_img,[INPUT_SIZE,INPUT_SIZE],method=0)
#     hist_img=tf.image.resize_images(hist_img,[INPUT_SIZE,INPUT_SIZE],method=0)
#     plt.imshow(curr_img.eval())
#     plt.imshow(hist_img.eval())    
#     plt.show()
#     mutex.acquire()
#     with tf.Session().as_default():
#     mutex.release() 
    curr_img= cv2.imread(os.path.join(rootDir,currImage),0)
    hist_img = cv2.imread(os.path.join(rootDir,histImage),0)
    curr_img = cv2.resize(curr_img,(INPUT_SIZE,INPUT_SIZE),interpolation=cv2.INTER_LINEAR)
    hist_img = cv2.resize(hist_img,(INPUT_SIZE,INPUT_SIZE),interpolation=cv2.INTER_LINEAR)
#     cv2.namedWindow('1',0)
#     cv2.namedWindow('2',0)
#     cv2.imshow('1',curr_img)
#     cv2.imshow('2',hist_img)
#     cv2.waitKey()
    curr_img=np.reshape(curr_img, [INPUT_SIZE,INPUT_SIZE,1])
    hist_img=np.reshape(hist_img, [INPUT_SIZE,INPUT_SIZE,1])
    curr_img_str=curr_img.tostring()
    hist_img_str=hist_img.tostring()
    
    example = tf.train.Example(features = tf.train.Features(
                        feature={
                                        'label':_int64_feature(int(label)),
                                        'curr_img':_bytes_feature(curr_img_str),
                                        'hist_img':_bytes_feature(hist_img_str)}))
    return example
    

def mutithread_generate_TFRecord(image_list,tfrecord_name_base,threadID,):
    num_images=0
    num_shard=0
    isUpdateTfrecordName=True
    for image_pair in image_list:
        if isUpdateTfrecordName:
            isUpdateTfrecordName=False
            tfrecord_name=tfrecord_name_base+str(threadID)+'_'+str(num_shard)                        
            writer = tf.python_io.TFRecordWriter(tfrecord_name)    
        rootDir = image_pair[2]
        example=convert_image_examples(os.path.join(INPUT_DATA_DIR,rootDir),image_pair[0], image_pair[1],image_pair[2])
        writer.write(example.SerializeToString())        
    
        if not (num_images+1)%500:
            writer.close()
            num_shard+=1 
            isUpdateTfrecordName = True  
            print(tfrecord_name+' is generated')
        num_images+=1
    else:
        if not isUpdateTfrecordName:
            writer.close()
            print(tfrecord_name+' is generated')
     

def convert_data_TFRecord(image_list,tfrecord_path):
    isFileExist(tfrecord_path)
#     tfrecord_writer=tf.python_io.TFRecordWriter()
    for category in CATELOGS:
        tfrecord_name_base=os.path.join(tfrecord_path,'data.'+category+'.tfrecord_')
        imagesInfo=image_list[category]
        imageNum = len(imagesInfo)
        imageNumPerThread = imageNum//NUM_THREAD
        thread_list=[]
        for num in range(NUM_THREAD):
            if num== NUM_THREAD-1:
                sub_image_list = imagesInfo[num*imageNumPerThread:imageNum] 
            else:
                sub_image_list = imagesInfo[num*imageNumPerThread:(num+1)*imageNumPerThread]
            thread = threading.Thread(target=mutithread_generate_TFRecord,args=(sub_image_list,tfrecord_name_base,num))
            thread_list.append(thread)
        for num in range(NUM_THREAD):   
            thread_list[num].setDaemon(True)
            thread_list[num].start()
        for num in range(NUM_THREAD):
            thread_list[num].join()
          
def main(_):    
#     config = tf.ConfigProto(device_count={"CPU": 4}, 
#                             inter_op_parallelism_threads = 1, 
#                             intra_op_parallelism_threads = 4,
#                             log_device_placement=True)
# 
#     tf.InteractiveSession(config=config)
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    convert_data_TFRecord(image_lists, OUTPUT_TFRECORD_DIR)
    print('all images are converted to TFRecords')
                
if __name__ == '__main__' :
    tf.app.run()