import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random

directory = "./data/"
correction = 0.2

def readImagepaths(skipHeader = True):
    lines=[]    
    with open(directory + "driving_log.csv",'r') as csvfile:
        reader = csv.reader(csvfile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def image_path(data_dir, path):
    return directory + "IMG/" + (path.split('/')[-1])
            
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
               
            for batch in batch_samples:
                center_image = read_image(image_path(directory, batch[0]))
                left_image = read_image(image_path(directory, batch[1]))
                right_image = read_image(image_path(directory, batch[2]))
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                
                center_angle = float(batch[3])
                # create adjusted steering measurements for the side camera images
                angles.append(center_angle)
                angles.append(center_angle+correction)
                angles.append(center_angle-correction)
                #flip images 
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)
                images.append(cv2.flip(left_image, 1))
                angles.append((center_angle+correction) * -1.0)
                images.append(cv2.flip(right_image, 1))
                angles.append((center_angle-correction) * -1.0)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


lines = readImagepaths()

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

model = Sequential()
#Preprocess incoming data, centered around zero with small standard deviation
#Cropping the images and focus on the road  
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,
                    validation_steps=len(validation_samples), epochs=3,verbose=1)

model.save('model.h5')
print ("model saved")
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

