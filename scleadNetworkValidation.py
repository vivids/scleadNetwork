'''
Created on Oct 10, 2018

@author: deeplearning
'''

import tensorflow as tf
import constants as ct
from scleadNetworkArchitecture import foward_propagation
from readImageFromTFRecord import readImageFromTFRecord
from writeAndReadFiles import readInfoFromFile
import time

def validate_network():
    dataSetSizeList = readInfoFromFile(ct.INFORMATION_PATH)
    validation_image_num = int(dataSetSizeList['validation'])
    image_inputs=tf.placeholder(tf.float32, (1,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL*2), 'validation_inputs')
    label_inputs =tf.placeholder(tf.float32,(1,ct.CLASS_NUM), 'validation_outputs')

    nn_output = foward_propagation(image_inputs,is_training=False)
    label_value_tensor = tf.argmax(label_inputs,1)
    pred_value_tensor = tf.argmax(nn_output,1)
#     correct_prediction = tf.equal(tf.argmax(nn_output,1), tf.argmax(label_inputs,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
   
    image_tensor,label_tensor= readImageFromTFRecord(ct.CATELOGS[2])
    saver = tf.train.Saver()
    with tf.Session() as sess :
         
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while(True):
            positive_sample_num=0
            negative_sample_num=0
            p2p = 0
            n2n=0
            ckpt = tf.train.get_checkpoint_state(ct.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                for _ in range(validation_image_num):
                    test_image, test_label = sess.run([image_tensor,label_tensor])
                    pred,label = sess.run([pred_value_tensor,label_value_tensor], feed_dict= {image_inputs:[test_image],label_inputs:[test_label]})
                    if label[0]:
                        positive_sample_num+=1
                        if pred[0]:
                            p2p+=1
                    else:
                        negative_sample_num+=1
                        if not pred[0]:
                            n2n+=1
                print(positive_sample_num)
                print(negative_sample_num)       
                correct_num = p2p+ n2n
                accuracy_score = correct_num/(positive_sample_num+negative_sample_num)
                p2p_score = p2p/positive_sample_num
                n2n_score = n2n/negative_sample_num
                print('after %s iteration, the validation accuracy is %g,p2p_score is %g,n2n_score is %g'%(global_step,accuracy_score,p2p_score,n2n_score))
            else:
                print('no model')
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             print(sess.run(update_ops))
            print('running..........')
            time.sleep(200)
        coord.request_stop()
        coord.join(threads) 
                           
if __name__ == '__main__':
    with tf.device('/cpu:0'):            
        validate_network()      
                                                              
