'''
Created on Oct 12, 2018

@author: deeplearning
'''

import tensorflow as tf
import constants as ct
from scleadNetworkArchitecture import foward_propagation
from readImageFromTFRecord import readImageFromTFRecord
from writeAndReadFiles import readInfoFromFile
from loadImageAndConvertToTFRecord import loadImageAndConvertToTFRecord

def test_network():
    if not ct.TEST_PERCENTAGE:
        loadImageAndConvertToTFRecord(test_percentage=100,validation_percentage=0,inputDataDir=ct.TEST_DATASET_PATH,
                                      infoSavePath=ct.TEST_INFOMATION_PATH,tfrecordPath=ct.TEST_TFRECORD_DIR)
        dataSetSizeList = readInfoFromFile(ct.TEST_INFOMATION_PATH)
    else:
        dataSetSizeList = readInfoFromFile(ct.INFORMATION_PATH)
    test_image_num = int(dataSetSizeList['testing'])
    image_inputs=tf.placeholder(tf.float32, (1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL*2), 'testing_inputs')
    label_inputs =tf.placeholder(tf.float32,(1,ct.CLASS_NUM), 'testing_outputs')

    nn_output = foward_propagation(image_inputs,is_training=False)
    correct_prediction = tf.equal(tf.argmax(nn_output,1), tf.argmax(label_inputs,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
   
    image_tensor,label_tensor= readImageFromTFRecord(ct.CATELOGS[1],tfrecord_dir= ct.TEST_TFRECORD_DIR)
    saver = tf.train.Saver()
    with tf.Session() as sess :
         
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        correct_prediction_list = []   
        ckpt = tf.train.get_checkpoint_state(ct.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for _ in range(test_image_num):
#             for _ in range(800):
                test_image, test_label = sess.run([image_tensor,label_tensor])
                per_correct_prediction = sess.run(correct_prediction, feed_dict= {image_inputs:[test_image],label_inputs:[test_label]})
                correct_prediction_list.append(per_correct_prediction[0])
            correct_num = 0 
            for rst in correct_prediction_list:
                correct_num+=rst
            accuracy_score = correct_num/len(correct_prediction_list)
            print('after %s iteration, the testing accuracy is %g'%(global_step,accuracy_score))
        else:
            print('no model')
        coord.request_stop()
        coord.join(threads) 
                           
if __name__ == '__main__':
    test_network()      
                                                              
