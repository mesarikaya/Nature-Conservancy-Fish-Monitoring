# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import glob
import pickle
from keras.preprocessing.image import ImageDataGenerator

# Function to import the images
url = 'C:/Users/MustafaErgin/Desktop/Kaggle Competitions/The Nature Convervancy Fisheries Monitoring'
fish_types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
def getImage(url = url, sub_folders = None, type = 'train'):
    """ This function retrieves the images from a specified path and subfolder.
    If there is not subfolder, the pictures under the provided path are done. 
    If there are subfolders, the pictures under the relevant folders are
    received"""
    if type == 'train':
        sub = 'Train data'
    else:
        sub = 'Test data'
    
    # initialioze the dataset
    data = []
    data2 = []
    if sub_folders != None:
        # Get Training data
        for folder in sub_folders:
            path = os.path.join('..',url+'/'+sub+'/'+folder+ '/')
            files_in_folder = [files.name for files in os.scandir(path) if files.is_file()]
            for img in files_in_folder:
                print("Image is: ", img, os.path.basename(os.path.normpath(path)))
                row = cv2.imread(path+img,1)
                resized = cv2.resize(row, (64, 64), interpolation = cv2.INTER_LINEAR)
                data.append(resized)
                data2.append(os.path.basename(os.path.normpath(path)))           
        return data, data2
    else: 
        # Get test data
        path = os.path.join('..',url+'/'+sub+'/')
        files_in_folder = [files.name for files in os.scandir(path) if files.is_file()]
        for img in files_in_folder:
            print("Image is: ", str(img), os.path.basename(os.path.normpath(path)))
            row = cv2.imread(path+img,1)
            resized = cv2.resize(row, (64, 64), interpolation = cv2.INTER_LINEAR)
            data.append([str(img), resized])
        return data
    
# Function to save the datasets to local drive
python_data_url = url + "/" + 'Python Data' + '/'
def save_to_local(files, file_names, dest = python_data_url):
    path = os.path.join('..',python_data_url)
    i = 0;
    for file in files:
        if not os.path.exists(dest):
            os.makedirs(dest)
        else:
            pickle.dump(file, open(os.path.join(path, str(file_names[i])+'.pkl'),'wb'), protocol = 4)
        i += 1

def load_from_local(file_name, dest = python_data_url):
    path = os.path.join('..',python_data_url)
    print(os.path.join(path,str(file_name)+'.pkl'))
    result = pickle.load(open(os.path.join(path,str(file_name)+'.pkl'),'rb'))
    return result

# Part 1 - Building the CNN
def createModel(epochs =1000):
        # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras import metrics
    
    # Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a third convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dense(output_dim = 65, activation = 'relu'))
    classifier.add(Dense(output_dim = 33, activation = 'relu'))
    classifier.add(Dense(output_dim = 8, activation = 'softmax'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [metrics.mae, metrics.categorical_accuracy])
    
    classifier.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                        samples_per_epoch=len(X_train), nb_epoch = epochs)


# Predicting the Test set results
y_pred = classifier.predict(test_datagen)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)