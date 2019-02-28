#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:52:00 2019

@author: wzhan
"""

import numpy as np
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import models 
from keras import layers 
#from keras import optimizers
#import os
#from PIL import Image


#%%
# include_top means not to include the last two fully connnected layers which act as the classifier. 
res_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(473,473,3))
#res_conv.summary()

## Freeze the layers except the last 4 layers 
#for layer in res_conv.layers[:-4]:
#    layer.trainable= False
#    
#for layer in res_conv.layers:
#    print(layer, layer.trainable)
#    
# Create a new model 
model = models.Sequential()

# Add the vgg convolutional base model
model.add(res_conv)

# Add new layers 
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu', input_dim=512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


#%%
# Divide the training dataset.
# It will have train and validation folders. And each folder should contain 3 folders(classes)

train_dir='/home/wzhan/Panicle/dataset/train'
validation_dir='/home/wzhan/Panicle/dataset/validation'
num_Train=580
num_Val=150
batch_size=1
train_steps=int(num_Train/batch_size)
val_steps=int(num_Val/batch_size)

#%%

# Crop the image according to your problem
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

# Create a generator when you crop the image 
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

# Compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model 
history = model.fit_generator(train_crops,
                    epochs=20,
                    steps_per_epoch=train_steps,
                    validation_data=validation_crops,
                    validation_steps=val_steps)

# Save the model 
model.save('panicle_resnet.h5')
