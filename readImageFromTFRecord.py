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
    curr_img = tf.decode_raw(features['curr_img'],tf.uint8)
    hist_img = tf.decode_raw(features['hist_img'],tf.uint8)
    labels = tf.decode_raw(features['label'],tf.uint8)
    curr_img=tf.reshape(curr_img,[ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    hist_img = tf.reshape(hist_img,[ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL])
    labels = tf.reshape(labels, [ct.CLASS_NUM])
    return curr_img,hist_img,labels

def conbineCurrAndHist2channelImage(curr_img,hist_img):
    return tf.concat([curr_img,hist_img],2) 
    
def combine_image_batch(curr_img,hist_img,label):
    image = conbineCurrAndHist2channelImage(curr_img,hist_img)
    capacity = ct.MIN_AFTER_DEQUEUE+3*ct.BATCH_SIZE
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=ct.BATCH_SIZE,capacity=capacity,
                                                     min_after_dequeue=ct.MIN_AFTER_DEQUEUE)
    return image_batch,label_batch
    

def readImageFromTFRecord():
    image_tfrecords = tf.train.match_filenames_once(os.path.join(ct.OUTPUT_TFRECORD_DIR,'data.training.tfrecord*'))
    image_queue = tf.train.string_input_producer(image_tfrecords,shuffle=True)
    image_batch_reader = tf.TFRecordReader()
    _,serialized_example = image_batch_reader.read(image_queue)
    curr_img,hist_img,labels=parse_examples(serialized_example)
    
    image_batch,label_batch = combine_image_batch(curr_img,hist_img,labels)
    return image_batch,label_batch


     
