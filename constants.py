'''
Created on Oct 8, 2018

@author: deeplearning
'''
INPUT_DATA_DIR ='/home/deeplearning/datasets/alarmClassification/experiment'
OUTPUT_TFRECORD_DIR = 'output/tfrecord'
MODEL_SAVE_PATH = 'output/model'
MODEL_NAME = 'sclead_network_model.ckpt'
INFORMATION_PATH='output/info'
CATELOGS = ('training','testing','validation')
TEST_PERCENTAGE = 0
VALIDATION_PERCENTAGE = 10
RESIZE_SIZE =360
INPUT_SIZE=[[256,256],[180,360],[360,180]]
INPUT_SIZE_CURR=INPUT_SIZE[0]
IMAGE_CHANNEL =1
NUM_THREAD=4
MIN_AFTER_DEQUEUE = 5000
BATCH_SIZE = 96
CLASS_NUM =2
LEARNING_RATE_INIT = 0.01
LEARNING_DECAY_RATE = 0.99
STEPS=50000

if not TEST_PERCENTAGE:
    TEST_DATASET_PATH ='/home/deeplearning/datasets/alarmClassification/test'
    TEST_INFOMATION_PATH = 'output/testInfo'
    TEST_TFRECORD_DIR = 'output/tfrecord_test'
    
#About architecture
# BLOCK1=[(256,64,1)]*1+[(256,64,2)]
# BLOCK2=[(512,128,1)]*1+[(512,128,2)]
# BLOCK3=[(1024,256,1)]*1+[(1024,256,2)]
# BLOCK4=[(2048,512,1)]*2

# BLOCK1=[(256,64,1)]*2+[(256,64,2)]
# BLOCK2=[(512,128,1)]*3+[(512,128,2)]
# BLOCK3=[(1024,256,1)]*5+[(1024,256,2)]
# BLOCK4=[(2048,512,1)]*3

# BLOCK1=[(256,64,1)]*2+[(256,64,2)]
# BLOCK2=[(512,128,1)]*2+[(512,128,2)]
# BLOCK3=[(1024,256,1)]*3+[(1024,256,2)]
# BLOCK4=[(2048,512,1)]*3

# BLOCK1=[(150,50,1)]*2+[(150,50,2)]
# BLOCK2=[(300,100,1)]*3+[(300,100,2)]
# BLOCK3=[(600,200,1)]*22+[(600,200,2)]
# BLOCK4=[(1200,400,1)]*3

# BLOCK1=[(256,64,1)]*2+[(256,64,2)]
# BLOCK2=[(512,128,1)]*3+[(512,128,2)]
# BLOCK3=[(1024,256,1)]*10+[(1024,256,2)]
# BLOCK4=[(2048,512,1)]*3


BLOCK1=[(150,50,1)]*2+[(150,50,2)]
BLOCK2=[(300,100,1)]*2+[(300,100,2)]
BLOCK3=[(600,200,1)]*2+[(600,200,2)]
BLOCK4=[(1200,400,1)]*1




