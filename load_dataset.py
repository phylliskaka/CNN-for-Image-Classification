#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:02:19 2018

@author: phyllis
"""

import numpy as np 
import os 
import cv2

#skitlearn 
from sklearn.cross_validation import train_test_split 

#%%

def load_dataset():
    """
    output: X_train, X_test, Y_train, Y_test 
    """
    def get_img(data_path):
        """
        read the image and transfer it to an array like object
        """
        img = cv2. imread(data_path)
        img = cv2.resize(img, (64, 64))
        return img
    
    path1 = '/home/phyllis/Desktop/ResNet50/input_dataset' #dataset path 
    
    listing = os.listdir(path1)
    m = len(listing)
    
    X = np.ndarray((m, 64, 64, 3)) # store the image matrix 
    Y = np.ndarray((m, 1)) #store the image label 
    
    for i in range(m):
        im = get_img(path1 + '/' + listing[i])
        X[i] = im
        Y[i] = (int(listing[i][0])) #file name is "label_index" like, retrieve the label from file name
    classes = int(max(Y)[0] + 1) # store the image classes number, in this project classes =10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0)
        
    return X_train, X_test, Y_train, Y_test, classes 
    



    