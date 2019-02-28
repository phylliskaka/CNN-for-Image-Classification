#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:25:21 2019

@author: wzhan
"""

'''
Transfer Learning using pre-trained models 
'''
#%%
import numpy as np
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import models 
from keras import layers 
#from keras import optimizers
#import os
#from PIL import Image


#%%
# include_top means not to lead the last two fully connnected layers which act as the classifier. 
# the last layer has a shape of 7*7*512
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(473,473,3))
#res_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(473,473,3))
#vgg_conv.summary()

# Divide the training dataset.
# It will have train and validation folders. And each folder should contain 3 folders(classes)

train_dir='/home/wzhan/Panicle/dataset/train'
validation_dir='/home/wzhan/Panicle/dataset/validation'
num_Train=580
num_Val=150
batch_size=1

#%%

# Crop the image so it dont has vertical wall and below part of plant.
def plant_crop(img, start_row, start_col, end_row, end_col):
    # img: a single image with channle_last
    # start_row: x of the start point of image 
    # start_col: y of the start point of image
    # end_row: x of the end point of image
    # end_col: y of the end point of image
    # crop_point = (0,int(width*0.23),int(height*0.86),int(width*0.78))
    
#    height, width = img.shape[0], img.shape[1]
    cropped = img[start_row:end_row, start_col:end_col]
#    #resize the picture 
#    cropped_im = Image.fromarray(cropped.astype('uint8'))
#    cropped_im.save(os.path.join('/Users/wzhan/Desktop/Cropped_im/train/panicle','crop.png'))
    return cropped

def crop_generator(batches, crop_point):
    # crop_point: a tuple of 4 float number, including following
    # start_row: x of the start point of image 
    # start_col: y of the start point of image
    # end_row: x of the end point of image
    # end_col: y of the end point of image
    # crop_point = (0,int(width*0.23),int(height*0.86),int(width*0.78))
    start_row, start_col, end_row, end_col = crop_point
    crop_height = end_row - start_row
    crop_width = end_col - start_col
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0],crop_height, crop_width, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = plant_crop(batch_x[i], start_row, start_col, end_row, end_col)
        yield (batch_crops, batch_y)

#%%
# ImageDataGenerator will generate batches of tensor image data with real-time data augmentation, augmenting data with 
# horizontal flip
train_datagen=ImageDataGenerator(rescale=1./255, horizontal_flip=True) 

validation_datagen=ImageDataGenerator(rescale=1./255)

train_features = np.zeros(shape=(num_Train, 14, 14, 512))
train_labels = np.zeros(shape=(num_Train, 1))
validation_features = np.zeros(shape=(num_Val, 14, 14, 512))
validation_labels = np.zeros(shape=(num_Val, 1))


# flow_from_diretory is a method of ImageDataGenerator which transfer image of direcotry to 
# tensor image data. It will resize the data, splite into batch size and you can also save_to_dir 
# which can save the augmented data to a dir path.  
train_batches = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(550,860),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

train_crops = crop_generator(train_batches, (0, int(860*0.23), int(550*0.86), int(860*0.78)))


validation_batches = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(550,860),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

validation_crops = crop_generator(validation_batches, (0, int(860*0.23), int(550*0.86), int(860*0.78)))

#%%

# calculate the features(generated by conv) of all the images, and get the train label. 
# Then features will be the X, train label will be the Y. It will feed into the following fully connected layer 
i=0 
for input_batch, labels_batch in train_crops: 
    features_batch = vgg_conv.predict(input_batch)
    train_features[i * batch_size:(i+1) * batch_size]= features_batch 
    train_labels[i * batch_size:(i+1) * batch_size]= labels_batch
    i +=1
    if i*batch_size >=num_Train:
        break

train_features = np.reshape(train_features, (num_Train, 14*14*512)) # flatten the feature

i=0 
for input_batch, labels_batch in validation_crops: 
    features_batch = vgg_conv.predict(input_batch)
    validation_features[i * batch_size:(i+1) * batch_size]= features_batch 
    validation_labels[i * batch_size:(i+1) * batch_size]= labels_batch
    i +=1
    if i*batch_size >=num_Train:
        break

validation_features = np.reshape(validation_features, (num_Val, 14*14*512)) # flatten the feature


#%%
# Create your own model(a simple feedforward network with 'relu' output layer)
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=14*14*512))
model.add(layers.Dense(256, activation='relu', input_dim=512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(train_features,
                    train_labels,
                    epochs=50,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

# how to visual the accuracy, Can i see from tensorboard ? 
# or I need to print out the accuracy after training  
    
    
model.save('seed_classification.h5')
    
    
    
    
    
    
    
    
    
