'''
Created on Oct 8, 2018

@author: deeplearning
'''
import tensorflow as tf
import os
import constants as ct
import cv2
def parse_examples(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
                        'label':tf.FixedLenFeature([], tf.string),
                        'curr_img':tf.FixedLenFeature([],tf.string),
                        'hist_img':tf.FixedLenFeature([],tf.string),
                        'row':tf.FixedLenFeature([],tf.int64),
                        'col':tf.FixedLenFeature([],tf.int64)})
    curr_img = tf.decode_raw(features['curr_img'],tf.uint8)
    hist_img = tf.decode_raw(features['hist_img'],tf.uint8)
    labels = tf.decode_raw(features['label'],tf.float32)
    row = tf.cast(features['row'],tf.int32)
    col = tf.cast(features['col'],tf.int32)


    curr_img=tf.reshape(curr_img,[ct.RESIZE_SIZE,ct.RESIZE_SIZE,ct.IMAGE_CHANNEL])
    hist_img = tf.reshape(hist_img,[ct.RESIZE_SIZE,ct.RESIZE_SIZE,ct.IMAGE_CHANNEL])
    labels = tf.reshape(labels, [ct.CLASS_NUM])
    proportion = row/col
    return curr_img,hist_img,labels,proportion

def conbineCurrAndHist2channelImage(curr_img,hist_img):
    return tf.concat([curr_img,hist_img],2)
    
def combine_image_batch(image,label):
    capacity = ct.MIN_AFTER_DEQUEUE+3*ct.BATCH_SIZE
    image_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                     batch_size=ct.BATCH_SIZE,num_threads=4,capacity=capacity,
                                                     min_after_dequeue=ct.MIN_AFTER_DEQUEUE)
    return image_batch,label_batch
 
def image_standardization(img):
    img = tf.cast(img, tf.float32)
    img = img/255.0
    return img
#     return tf.image.per_image_standardization(img)

def image_preprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)   
    return image        

def readImageFromTFRecord(category,shuffle =False,num_epochs=None,tfrecord_dir=ct.OUTPUT_TFRECORD_DIR):
    image_tfrecords = tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'.tfrecord*'))
    image_reader = tf.TFRecordReader()
    image_queue = tf.train.string_input_producer(image_tfrecords,shuffle =shuffle,num_epochs=num_epochs)
    _,serialized_example = image_reader.read(image_queue)
    curr_img,hist_img,labels,proportion=parse_examples(serialized_example) 
    
    curr_img = image_standardization(curr_img)    
    hist_img = image_standardization(hist_img) 
    image = conbineCurrAndHist2channelImage(curr_img,hist_img)
    return image,labels,proportion
    


def readImageBatchFromTFRecord(category):
    image,labels,_=readImageFromTFRecord(category,shuffle =True,num_epochs=None)

#     with tf.variable_scope('global_step',reuse=tf.AUTO_REUSE):
#         global_step = tf.Variable(0, trainable=False)
#     selected_input_entrance = global_step%3
#     
#     image_256_256 = tf.image.resize_images(image,[256,256],method=0)
#     image_180_360 = tf.image.resize_images(image,[180,360],method=0)
#     image_360_180 = tf.image.resize_images(image,[360,180],method=0)
#     
#     img = tf.cond(tf.equal(selected_input_entrance,0),lambda:image_256_256,
#                     lambda:tf.cond(tf.equal(selected_input_entrance,1),lambda:image_180_360,lambda:image_360_180))
    image= image_preprocess(image)
    image_batch,label_batch = combine_image_batch(image,labels)

    return image_batch,label_batch

