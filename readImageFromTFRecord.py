'''
Created on Oct 8, 2018

@author: deeplearning
'''
import tensorflow as tf
import os
import constants as ct

def parse_examples(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
                        'label':tf.FixedLenFeature([], tf.string),
                        'curr_img':tf.FixedLenFeature([],tf.string),
                        'hist_img':tf.FixedLenFeature([],tf.string)})
    curr_img = tf.decode_raw(features['curr_img'],tf.float32)
    hist_img = tf.decode_raw(features['hist_img'],tf.float32)
    labels = tf.decode_raw(features['label'],tf.float32)
    curr_img=tf.reshape(curr_img,[ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    hist_img = tf.reshape(hist_img,[ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    labels = tf.reshape(labels, [ct.CLASS_NUM])
    return curr_img,hist_img,labels

def conbineCurrAndHist2channelImage(curr_img,hist_img):
    return tf.concat([curr_img,hist_img],2) 
    
def combine_image_batch(image,label):
    capacity = ct.MIN_AFTER_DEQUEUE+3*ct.BATCH_SIZE
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=ct.BATCH_SIZE,capacity=capacity,
                                                     min_after_dequeue=ct.MIN_AFTER_DEQUEUE)
    return image_batch,label_batch
    

def readImageFromTFRecord(category,shuffle =False,num_epochs=None,tfrecord_dir=ct.OUTPUT_TFRECORD_DIR):
    image_tfrecords = tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'.tfrecord*'))
    image_reader = tf.TFRecordReader()
    image_queue = tf.train.string_input_producer(image_tfrecords,shuffle =shuffle,num_epochs=num_epochs)
    _,serialized_example = image_reader.read(image_queue)
    curr_img,hist_img,labels=parse_examples(serialized_example)      
    image = conbineCurrAndHist2channelImage(curr_img,hist_img)
    return image,labels
    


def readImageBatchFromTFRecord(category):
    image,labels=readImageFromTFRecord(category,shuffle =True,num_epochs=None)
    image_batch,label_batch = combine_image_batch(image,labels)
    return image_batch,label_batch

