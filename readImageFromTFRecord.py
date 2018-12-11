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
    
def combine_image_batch(info,flag):
    capacity = ct.MIN_AFTER_DEQUEUE+3*ct.BATCH_SIZE
    
    image_batch_256_256,label_batch_256_256,proportion_batch_256_256 = tf.train.shuffle_batch(info['info_256_256'],
                                         batch_size=ct.BATCH_SIZE,num_threads=4,capacity=capacity,
                                         min_after_dequeue=ct.MIN_AFTER_DEQUEUE,shared_name='shape_256_256',name='shape_256_256')
    image_batch_180_360,label_batch_180_360,proportion_batch_180_360 = tf.train.shuffle_batch(info['info_180_360'],
                                     batch_size=ct.BATCH_SIZE,num_threads=4,capacity=capacity,
                                     min_after_dequeue=ct.MIN_AFTER_DEQUEUE,shared_name='shape_180_360',name='shape_180_360')
    image_batch_360_180,label_batch_360_180,proportion_batch_360_180 = tf.train.shuffle_batch(info['info_360_180'],
                                     batch_size=ct.BATCH_SIZE,num_threads=4,capacity=capacity,
                                     min_after_dequeue=ct.MIN_AFTER_DEQUEUE,shared_name='shape_360_180',name='shape_360_180')
    
    image_batch,label_batch,proportion_batch = tf.cond(tf.equal(flag,0),lambda:[image_batch_256_256,label_batch_256_256,proportion_batch_256_256],lambda:tf.cond(tf.equal(flag,1),
                                                                                                                                                                 lambda:[image_batch_180_360,label_batch_180_360,proportion_batch_180_360],
                                                                                                                                                                 lambda:[image_batch_360_180,label_batch_360_180,proportion_batch_360_180]))

    
    return image_batch,label_batch,proportion_batch



 
def image_standardization(img):
    img = tf.cast(img, tf.float32)
    img = img/255.0
    return img
#     return tf.image.per_image_standardization(img)

def image_preprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)   
    return image        

def readImageFromTFRecordForTest(category, shuffle =False,num_epochs=None,tfrecord_dir=ct.OUTPUT_TFRECORD_DIR):
    image_tfrecords= tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'*'))    
    image_queue = tf.train.string_input_producer(image_tfrecords,shuffle =shuffle,num_epochs=num_epochs)
    image_reader = tf.TFRecordReader(name='image_reader_test') 
    _,serialized_example = image_reader.read(image_queue)
    curr_img,hist_img,label,proportion=parse_examples(serialized_example)
    curr_img = image_standardization(curr_img)    
    hist_img = image_standardization(hist_img)  
    image = conbineCurrAndHist2channelImage(curr_img,hist_img)    

    return image,label,proportion

def readImageFromTFRecord(category, shuffle =False,num_epochs=None,tfrecord_dir=ct.OUTPUT_TFRECORD_DIR):
    image_tfrecords_180_360 = tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'.shape_180_360*'))
    image_tfrecords_360_180 = tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'.shape_360_180*'))
    image_tfrecords_256_256 = tf.train.match_filenames_once(os.path.join(tfrecord_dir,'data.'+category+'.shape_256_256*'))    
#     image_tfrecords = tf.cond(tf.equal(flag,0),lambda:image_tfrecords_256_256,lambda:tf.cond(tf.equal(flag,1),lambda:image_tfrecords_180_360,lambda:image_tfrecords_360_180))
    

#     image_queue = tf.train.string_input_producer(image_tfrecords,shuffle =shuffle,num_epochs=num_epochs)
    image_queue_180_360 = tf.train.string_input_producer(image_tfrecords_180_360,shuffle =shuffle,num_epochs=num_epochs)
    image_queue_360_180 = tf.train.string_input_producer(image_tfrecords_360_180,shuffle =shuffle,num_epochs=num_epochs)
    image_queue_256_256 = tf.train.string_input_producer(image_tfrecords_256_256,shuffle =shuffle,num_epochs=num_epochs) 
#     image_queue = tf.cond(tf.equal(flag,0),lambda:image_queue_256_256,lambda:tf.cond(tf.equal(flag,1),lambda:image_queue_180_360,lambda:image_queue_360_180))

    image_reader_180_360 = tf.TFRecordReader(name='image_reader_180_360') 
    image_reader_360_180 = tf.TFRecordReader(name='image_reader_360_180') 
    image_reader_256_256 = tf.TFRecordReader(name='image_reader_256_256') 
        
    _,serialized_example_180_360 = image_reader_180_360.read(image_queue_180_360)
    _,serialized_example_360_180 = image_reader_360_180.read(image_queue_360_180)
    _,serialized_example_256_256 = image_reader_256_256.read(image_queue_256_256)
    
#     serialized_example = tf.cond(tf.equal(flag,0),lambda:serialized_example_256_256,lambda:tf.cond(tf.equal(flag,1),lambda:serialized_example_180_360,lambda:serialized_example_360_180))
#     curr_img,hist_img,labels,proportion=parse_examples(serialized_example)
    curr_img_180_360,hist_img_180_360,label_180_360,proportion_180_360=parse_examples(serialized_example_180_360)
    curr_img_360_180,hist_img_360_180,label_360_180,proportion_360_180=parse_examples(serialized_example_360_180) 
    curr_img_256_256,hist_img_256_256,label_256_256,proportion_256_256=parse_examples(serialized_example_256_256)  

#     curr_img = image_standardization(curr_img)    
#     hist_img = image_standardization(hist_img) 
    curr_img_180_360 = image_standardization(curr_img_180_360)    
    hist_img_180_360 = image_standardization(hist_img_180_360) 
    curr_img_360_180 = image_standardization(curr_img_360_180)    
    hist_img_360_180 = image_standardization(hist_img_360_180) 
    curr_img_256_256 = image_standardization(curr_img_256_256)    
    hist_img_256_256 = image_standardization(hist_img_256_256) 
#     image = conbineCurrAndHist2channelImage(curr_img,hist_img)    
    image_180_360 = conbineCurrAndHist2channelImage(curr_img_180_360,hist_img_180_360)
    image_360_180 = conbineCurrAndHist2channelImage(curr_img_360_180,hist_img_360_180)
    image_256_256 = conbineCurrAndHist2channelImage(curr_img_256_256,hist_img_256_256)
    
    image_180_360= image_preprocess(image_180_360)  
    image_360_180= image_preprocess(image_360_180) 
    image_256_256= image_preprocess(image_256_256) 
    
    info_180_360=[image_180_360,label_180_360,proportion_180_360]
    info_360_180=[image_360_180,label_360_180,proportion_360_180]
    info_256_256=[image_256_256,label_256_256,proportion_256_256]
    return {'info_180_360':info_180_360,'info_360_180':info_360_180,'info_256_256':info_256_256}
#     return image0,labels0,proportion0,image1,labels1,proportion1,image2,labels2,proportion2
    


def readImageBatchFromTFRecord(category,flag):
    info=readImageFromTFRecord(category,shuffle =True,num_epochs=None)
#     image= image_preprocess(image)  
    image_batch,label_batch,proportion_batch = combine_image_batch(info,flag)
 
    return image_batch,label_batch,proportion_batch

