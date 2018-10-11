'''
Created on Oct 8, 2018

@author: deeplearning
'''
INPUT_DATA_DIR ='/home/deeplearning/datasets/alarmClassification'
OUTPUT_TFRECORD_DIR = 'output/tfrecord'
MODEL_SAVE_PATH = 'output/model'
MODEL_NAME = 'sclead_network_model.ckpt'
INFORMATION_PATH='output/info'
CATELOGS = ('training','testing','validation')
TEST_PERCENTAGE = 0
VALIDATION_PERCENTAGE = 10
INPUT_SIZE=512
IMAGE_CHANNEL =1
NUM_THREAD=4
MIN_AFTER_DEQUEUE = 100
BATCH_SIZE = 16
CLASS_NUM =2
LEARNING_RATE_INIT = 0.01
LEARNING_DECAY_RATE = 0.99
STEPS=50000


BLOCK1=[(256,64,1)]*1+[(256,64,2)]
BLOCK2=[(512,128,1)]*1+[(512,128,2)]
BLOCK3=[(1024,256,1)]*1+[(1024,256,2)]
BLOCK4=[(2048,512,1)]*2
