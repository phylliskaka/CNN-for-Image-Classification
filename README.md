# ResNet50-for-image-recognition
The goal of this project is to implement a 50 layers Neural network with skip connection to do classification for image

## Dataset 

Dataset name: Turkey Ankara Ayrancı Anadolu High School's Sign Language Digits Dataset. [Download here](https://github.com/ardamavi/Sign-Language-Digits-Dataset#turkey-ankara-ayranc%C4%B1-anadolu-high-schools-sign-language-digits-dataset)  

Detailed information of dataset:    

|    Image size   |   Color space  |  Number of classes  | Number of paticipant students |  Number of sample per student |
|-----------------|:--------------:|--------------------:|:-----------------------------:|:-----------------------------:|
|     100x100     |     RGB        |    10 (Digits 0-9)  |            218                |          10                   |


Data Preview:     
![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/data_preview.png)

The data including 10 types of hand gesture to represent integer range from 0 to 9. The dataset contain 1800+ RGB images. Considering the speed of training, in this project, we only take 150 RGB images for training(each 15 images). The [reduced dataset](https://github.com/phylliskaka/ResNet50-for-image-recognition/tree/master/input_dataset) is avaible in the repository.

## ResNet50 introduction 

1. What and Why ResNet50    

Why ResNet: Gradient Vanishing   
The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However,very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. In backpropagation, gradient reduced to almost 0 when reach the first layer.   

![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/gradient%20vanishing.png) 

What is it: The data in layer l can skip directly added to layer l+2 (or l+3), called skip connection    

![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/residual%20block.png)   

2. Architecture of ResNet50   

![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/ResNet50.png)   

The architecture of Identity block and Convolutional block(two main type of block in resnet)is shown following:   
Identity block:   

![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/identity_block.png) 

Convolutional block:   

![alt text](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/image/convolutional_block.png)  

## Implementation in python using keras  
### Function need to implement before building ResNet50 model   

* [load_dataset()](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/load_dataset.py)      
   This function import the reduced dataset and resize the original image (100, 100, 3) to (64, 64, 3) and stack 150 images together to form a matrix X of shape (150, 64, 64, 3). Finally, it split the dataset into X_test and X_training. The label Y generated according to file name('classes_index' like)  
   
* [convert_to_one_hot_vector()](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/one_hot_vector.py)    This function transfer the label Y_train of size(120, 1) to size (120, 10). Each column is one hot vector. 

* [identity_block()](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/identity_block.py)    
   This function implement the identity block which will be used in building ResNet model.     
   
* [convolution_block()](https://github.com/phylliskaka/ResNet50-for-image-recognition/blob/master/conv_block.py)   
   This function implement the identity block which will be used in building ResNet model.     
   
### Bullding model 
#### Pseudo code    
#preprocessing dataset: Load dataset, Normalization, Convert Y to one hot vecotor     
#build ResNet50 model: stage 1-5    
#model fit on train data   
#prediction on test data   

Figure resource: https://hub.coursera-notebooks.org/user/sdqrefghxxwpvtltbjlhbz/notebooks/week2/ResNets/Residual%20Networks%20-%20v2.ipynb

