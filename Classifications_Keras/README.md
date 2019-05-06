# CNN for Image Classfication
This project aiming to classify a image into panicle or no panicle. 

## Dataset
The dataset is a private dataset. It includes 2570 images(200*200, RGB). Due to the limitation of very small dataset, we need to use the 
concept of Fine-tuning.   

## Model
We obtain the original model and trained weight from Keras Application (ResNet50, VGG16, InceptionV3). After replace the classifier head
of neural net model, we trained the model on the dataset with only last 4 layers of CNN and classifier trainable. 

## Result
VGG16 are able to achieve testing accuracy of 95.6%.    
ResNet50 are able to achieve testing accuracy of 88.3%.    
InceptionV3 are able to achieve testing accuracy of 80.3%.    

