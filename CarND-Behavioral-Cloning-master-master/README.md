# CarND-Behavioral-Cloning-master

# Behavioral Cloning Project
The goals / steps of this project are the following:
•	Use the simulator to collect data of good driving behavior
•	Build, a convolution neural network in Keras that predicts steering angles from images
•	Train and validate the model with a training and validation set
•	Test that the model successfully drives around track one without leaving the road
•	Summarize the results with a written report

# Model Architecture and Training Strategy
The model is a modification of the NVIDIA architecture with an additional dropout layer. The method to prevent over fitting is by adding a dropout layer in between the fully connected layers. The model uses Relu as the activation function to add up the nonlinearity. The model uses Adam Optimizer. It sets the batch size as 32 and epoch number as 4. After each epoch, 20% of data is split from the training dataset for the use of validation. 

Also, for each image, the model crops the top 50 pixels and bottom 20 pixels so that the car focuses on the road. Additionally, the model normalizes the pixel value to the range of [-0.5, 0.5] so that the dataset has 0 mean. The data incorporates the left, center and right cameras. The side cameras training data can significantly improve the vehicle’s ability in recovering from shifting to the side of the road.  Since the left, center and right cameras see the road differently with an offset angle, we add 0.2 as the correction factor. Specifically, we add 0.2 to the angle for left camera and subtract 0.2 from the angle for the right camera.  

In addition, the car only drives counter-clock wise during training, so we flip the images horizontally so that the car could drive clock wise. This could data augmentation technique can not only create more data for training, but also trains the car to drive in different orientation. The label angles for flip images needs to be reversed too. In order to prevent bias in dataset, the model shuffles the data before feeding into Neural Network.     

Layer 	Description 

Input 	160x320x3 color images 

Cropping 	Crop the top 50 pixels and bottom 20 pixels of each image 

Normalization	Normalize the images pixel to the range of [-0.5, 0.5]

Convolution 	Weights Shape: (24, 5, 5); Subsample: (2, 2); Activation: Relu 

Convolution 	Weights Shape: (36, 5, 5); Subsample: (2, 2); Activation: Relu 

Convolution 	Weights Shape: (48, 5, 5); Subsample: (2, 2); Activation: Relu 

Convolution 	Weights Shape: (64, 5, 5); Subsample: (2, 2); Activation: Relu 

Convolution 	Weights Shape: (64, 5, 5); Subsample: (2, 2); Activation: Relu 

Flatten 	

Fully Connected 	Output Shape: 100; Activation: Relu

Dropout 	Possibility to Drop: 0.5

Fully Connected	Output Shape: 50

Fully Connected	Output Shape: 10

Fully Connected	Output Shape: 1

Fully Connected 	Weights Shape: (84, 43); Output Shape: 43


