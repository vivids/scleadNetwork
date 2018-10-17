'''
Created on Oct 8, 2018

@author: deeplearning
'''
import os
import tensorflow as tf
import constants as ct
from readImageFromTFRecord import readImageBatchFromTFRecord
from loadImageAndConvertToTFRecord import loadImageAndConvertToTFRecord
from scleadNetworkArchitecture import foward_propagation


def train_network(training_image_num):
    global_step = tf.Variable(0, trainable=False)
    image_inputs=tf.placeholder(tf.float32, (ct.BATCH_SIZE,ct.INPUT_SIZE,ct.INPUT_SIZE,ct.IMAGE_CHANNEL*2), 'inputs')
    label_inputs =tf.placeholder(tf.float32,(ct.BATCH_SIZE,ct.CLASS_NUM), 'outputs')
    
    nn_output = foward_propagation(image_inputs)
#     nn_output = tf.clip_by_value(nn_output,1e-15,1.0)
#     nn_output = nn_output/50000
#     nn_softmax = tf.nn.softmax(nn_output)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nn_output, labels=tf.argmax(label_inputs,1))
    cross_entropy_loss_mean = tf.reduce_mean(cross_entropy_loss)
    loss_func = cross_entropy_loss_mean
    learning_rate = tf.train.exponential_decay(ct.LEARNING_RATE_INIT, global_step, training_image_num/ct.BATCH_SIZE, ct.LEARNING_DECAY_RATE)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func, global_step)
    
   

    image_batch_tensor,label_batch_tensor= readImageBatchFromTFRecord(ct.CATELOGS[0])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
      
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
         
        for i in range(ct.STEPS+1):
            image_batch, label_batch = sess.run([image_batch_tensor,label_batch_tensor])
            _,loss_val,step = sess.run([train_step, cross_entropy_loss_mean,global_step], 
                                       feed_dict= {image_inputs:image_batch,label_inputs:label_batch})
#             output = sess.run(nn_output, 
#                                        feed_dict= {image_inputs:image_batch,label_inputs:label_batch})    
#             print(output)    
            if not (i%100):
                print('after %d iteration, loss is %g'%(step,loss_val))
                if not (i%1000):
                    saver.save(sess, os.path.join(ct.MODEL_SAVE_PATH,ct.MODEL_NAME),global_step)
                  
        coord.request_stop()
        coord.join(threads)

def main(_):
    training_image_num=loadImageAndConvertToTFRecord()
#     training_image_num=3922
    train_network(training_image_num)

if __name__ == '__main__' :
    tf.app.run()