# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 19:12:08 2018

@author: lisuk
"""
import numpy as np
import os
from skimage import util
from skimage.io import imread
from skimage.transform import resize
import h5py
import matplotlib.pyplot as plt
import PIL.ImageOps 

data_path = './'

height = 228
width = 304

depth_height=128
depth_width=160
ch =3


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, height, width, ch), dtype=np.uint8)
    depths = np.ndarray((total, depth_height, depth_width), dtype=np.uint8)
    
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        
        if '.h5' in image_name:
            continue
        image_depth_name = image_name.split('.')[0] + '.h5'
        
        img = imread(os.path.join(train_data_path, image_name), as_grey = False)
        #img = rescale(img, 1.0 / 3.0, anti_aliasing = False)
        img = resize(img, (height,width), anti_aliasing = True)

        depth_read = h5py.File(os.path.join(train_data_path, image_depth_name), 'r')
        depth = depth_read.get('/depth')
        depth = np.array(depth, dtype=np.uint8)
        #depth = rescale(depth, 1.0 / 3.0, anti_aliasing = False)
        depth = resize(depth, (depth_height, depth_width), anti_aliasing = True)
        depth_read.close()
        
        imgs[i] = np.array([img])
        depths[i] = depth

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('./imgs_train.npy', imgs)
    np.save('./imgs_depth_train.npy', depths)
    print('Saving to .npy files done.')
    

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_depth_train = np.load('imgs_depth_train.npy')
    return imgs_train, imgs_depth_train


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, height, width, ch), dtype=np.uint8)
    depths = np.ndarray((total, depth_height, depth_width), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)
    img = []
    i = 0
    print('-'*30)
    print('Creating testing images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('_')[0])
        
        if '.h5' in image_name:
            continue
        image_depth_name = image_name.split('.')[0] + '.h5'
        
        img = imread(os.path.join(test_data_path, image_name), as_grey = False)
        img = resize(img, (height, width), anti_aliasing=True)
        
        depth_read = h5py.File(os.path.join(test_data_path, image_depth_name), 'r')
        depth = depth_read.get('/depth')
        depth = np.array(depth, dtype=np.uint8)

        depth = resize(depth, (depth_height, depth_width), anti_aliasing = True)
        #depth_read.close()

        imgs[i] = np.array([img])
        depths[i] = depth
        imgs_id[i] = img_id
        

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('./imgs_test.npy', imgs)
    np.save('./imgs_depth_test.npy', depths)
    np.save('./imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')
    
    
def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_depth_test = np.load('imgs_depth_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id, imgs_depth_test


if __name__ == '__main__':
    create_train_data()
    create_test_data()
