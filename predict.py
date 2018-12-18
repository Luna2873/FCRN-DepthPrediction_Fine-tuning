# -*- coding: utf-8 -*-
"""
@Original author: FCRN-DepthPrediction

Created on Fri Dec  7 23:28:22 2018
Modified by SL

"""

import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing as scale
from skimage import util 
from math import sqrt

import models

def predict(model_data_path, image_path, gt_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    
     # Read image
    gt = Image.open(gt_path)
    gt = gt.resize([160,128], Image.ANTIALIAS)
    gt = np.array(gt).astype('float32')

    min_max_scaler = scale.MinMaxScaler()
    gt = min_max_scaler.fit_transform(gt)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        print(pred.shape)
        
        a = pred[0,:,:,0]
        a = util.invert(a)
        a = min_max_scaler.fit_transform(a)
        
        print('RMS error.. the best value is 0.0')
        rms = sqrt(mean_squared_error(gt, a)) 
        print(rms)
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(a, cmap='gray', interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    parser.add_argument('gt_paths', help='Directory of depths to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths, args.gt_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()


# python predict.py ./NYU_FCRN-checkpoint/NYU_FCRN.ckpt test/4063046_0c36f21fe5_o.jpg
