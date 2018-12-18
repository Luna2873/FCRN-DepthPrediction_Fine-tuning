# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:31:42 2018

@author: lisuk
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

from datetime import datetime
from tensorflow.python.platform import gfile
from dataprocessing import load_train_data, load_test_data
from models.fcrn import ResNet50UpProj as Network


logs_path_train = './log/train/'
logs_path_test = './log/test/'

meta_path_restore = './NYU_FCRN-checkpoint/NYU_FCRN.ckpt.meta'
CKPTDIR ='./NYU_FCRN-checkpoint/'

weights_checkpoint_path = './newckpt/'

EPOCHS = 20

# Default input size
height = 228
width = 304
channels = 3
batch_size = 8
depth_height=128
depth_width=160


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

def tf_huber_loss(y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

def tf_mse_loss(y_true, y_pred):
        return tf.losses.mean_squared_error(y_true,y_pred)  

def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    ########################## Load Image ######################
    
    with tf.device('/cpu:0'):
        
        imgs_train, imgs_depth_train = load_train_data()
        imgs_train = imgs_train.astype('float32')
        
        imgs_depth_train = imgs_depth_train.astype('float32')
        imgs_depth_train = np.expand_dims(np.asarray(imgs_depth_train), axis = -1)
        
        imgs_test, imgs_id_test, imgs_depth_test = load_test_data()
        imgs_test = imgs_test.astype('float32')
        
        imgs_depth_test = imgs_depth_test.astype('float32')
        imgs_depth_test = np.expand_dims(np.asarray(imgs_depth_test), axis = -1)
        
    
    ########################## setupNet / placeholder / finetunLayer ######################
     
    images = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    depths = tf.placeholder(tf.float32, [None, depth_height, depth_width, 1])
    
    net = Network({'data': images}, batch_size, 1, is_training = True) # inputs, batch, keep_prob, is_training, trainable = True
    
    fine_tuing_layers = ['res4a','res4b', 'res4c','res4d','res4e', 'res4f']
    tunning_params = []
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    for W in tf.trainable_variables():
        if "batch_normalization" not in W.name:
            print(W.name)
        for layer in fine_tuing_layers:
            if layer in W.name:
                print('tune')
                tunning_params.append(W)
                break
    
    ########################## Loss / optimizer ######################
    with tf.name_scope('loss'):
        loss = tf_mse_loss(depths, net.get_output())
        tf.summary.scalar('huber_loss', loss)
   
     
    lr_tune = tf.train.exponential_decay(1e-5, global_step, 5000, 0.1, staircase=True)
    
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate = lr_tune).minimize(loss, global_step = global_step, var_list = tunning_params)
    
    ########################## RunOptions / GPU option ######################
    #run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    
    # Assume that you have 8GB of GPU memory and want to allocate ~2.6GB:
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
    
    ########################## Sesstion ######################
    with tf.Session(config = tf.ConfigProto(log_device_placement=True)) as sess:
        
        #sess.run(op, feed_dict = fdict, options = run_options)
        sess.run(tf.global_variables_initializer())
        
        learnable_params = tf.trainable_variables()
        # define savers
        saver_learnable = tf.train.Saver(learnable_params, max_to_keep=4)
        print('Saver....')
        
        # log
        merged = tf.summary.merge_all() 
        writer_train = tf.summary.FileWriter(logs_path_train, graph = sess.graph)
        writer_test = tf.summary.FileWriter(logs_path_test, graph = sess.graph)        

        
        # Load the converted parameters
        print('Loading the model...')
        
        _ckpt = tf.train.get_checkpoint_state(CKPTDIR)
        
        
        if _ckpt and _ckpt.model_checkpoint_path:
            # Use to load from ckpt file#        
            saver = tf.train.import_meta_graph(meta_path_restore) 
            print("Meta graph restored successfully!")
            print('-'*30)
            
            saver.restore(sess, _ckpt.model_checkpoint_path)
            print("Weights restored successfully!")
            print('-'*30)
        
        #learnable_params = tf.trainable_variables()
        
        # initialize the queue threads to start to shovel data
        print('Start Coord and Threads')
        print('-'*30)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print('-'*30)
        
        for epoch in range(EPOCHS):
            for i in range(1000):
                setp = epoch * 1000 + i
                
                start_time = time.time()

                _, loss_value, out_depth, train_summ = sess.run([train_op, loss, net.get_output(), merged], feed_dict={images:imgs_train, depths:imgs_depth_train}) # options = run_options          
                writer_train.add_summary(train_summ, setp)  #train log
                #print(net.get_output())
                
                duration = time.time() - start_time
    
                if i % 10 == 0:
                    # To log validation accuracy.
                    validation_loss, pred, valid_summ = sess.run([loss, net.get_output(), merged], feed_dict={images: imgs_test, depths:imgs_depth_test})
                    writer_test.add_summary(valid_summ, setp)
        
                    print('sec_per_batch')
                    sec_per_batch = float(duration)
                    print("%s: %d[epoch]: %d[iteration]: train loss = %.4f : valid loss = %.4f : %.3f [sec/batch]" % (datetime.now(), epoch, i, loss_value, validation_loss, sec_per_batch))
                    
        saver_learnable.save(sess, weights_checkpoint_path) 
                    
        # stop queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()