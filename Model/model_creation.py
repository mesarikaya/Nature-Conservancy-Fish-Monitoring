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
#import glob
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.svm import SVC
# Function to import the images
url = 'C:/Users/MustafaErgin/Desktop/Kaggle Competitions/The Nature Convervancy Fisheries Monitoring'
fish_types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
python_data_url = url + "/" + 'Python Data' + '/'

def getImage(url = url, sub_folders = None, type = 'train'):
    """ This function retrieves the images from a specified path and subfolder.
    If there is not subfolder, the pictures under the provided path are done. 
    If there are subfolders, the pictures under the relevant folders are
    received"""
    if type == 'train':
        sub = 'Train data'
    else:
        sub = 'Test2 data'
    
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
                resized = cv2.resize(row, (64, 64), interpolation = cv2.INTER_AREA)
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
            resized = cv2.resize(row, (64, 64), interpolation = cv2.INTER_AREA)
            data.append([str(img), resized])
        return data
    
# Function to save the datasets to local drive

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
def createCnnModel(X_train, Y_train, datagen, input_epochs=1000):
        # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras import metrics
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    from keras.optimizers import Adam, SGD
    from keras.callbacks import EarlyStopping
    from sklearn.metrics import log_loss
    from keras.layers.normalization import BatchNormalization
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, 
                                                          test_size=0.01, random_state=1)
    
    # Initialising the CNN
    classifier = Sequential()

    classifier.add(Dropout(0.4, input_shape=(64, 64, 3)))
    
    # Step 1 - Convolution
    classifier.add(Convolution2D(64, (3, 3), input_shape = (64, 64, 3), activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.4))
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(64, (3, 3), activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.4))
    
    # Adding a second convolutional layer
    classifier.add(Convolution2D(64, (3, 3), activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.4))
    
#    # Adding a second convolutional layer
#    classifier.add(Convolution2D(64, 3, 3, activation=PReLU(alpha_initializer='zero',weights=None)))
#    classifier.add(PReLU())
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))
#    classifier.add(Dropout(0.2))
    
    


    # Step 3 - Flattening
    classifier.add(Flatten())
    classifier.add(Dropout(0.4, input_shape=(128,)))
    # Step 4 - Full connection
    classifier.add(Dense(units = 89, activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 65, activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 40, activation=PReLU(alpha_initializer='zero',weights=None)))
    classifier.add(PReLU())
    classifier.add(Dropout(0.4))
#    classifier.add(Dense(output_dim = 50, activation=PReLU(alpha_initializer='zero',weights=None)))
#    classifier.add(PReLU())
#    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 8, activation = "softmax"))
    
    # Compiling the CNN
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),]
    classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = [metrics.categorical_accuracy])
    # from keras.callbacks import EarlyStopping
    # early_stopping = EarlyStoppisng(monitor='val_loss', patience=2)
    
    validation_generator = datagen.flow(X_valid, Y_valid, batch_size=100, seed=1)
    
    classifier.fit_generator(datagen.flow(X_train, Y_train, batch_size=100, seed =1),
                        steps_per_epoch=33, epochs = input_epochs,
                        validation_data=validation_generator,
                        callbacks=callbacks,
                        validation_steps=200)
    
    predictions_valid = classifier.predict(validation_generator.x.astype('float32'), batch_size=100, verbose=2)
    score = log_loss(validation_generator.y, predictions_valid)
    print('Score log_loss: ', score)
    return classifier

def createRFmodel(X, Y, X_test, datagen, n_trees = 500):
    # Splitting the dataset into the Training set and Test set
#    X = np.array(X)
#    Y= np.array(Y)

#    X_train, Y_train = X, Y
    m = datagen.flow(X, Y, batch_size=100, seed=1)
    X_train = m.x
    Y_train = m.y
    dataset_size = len(X_train)
    X_train_TwoDim = X_train.reshape(dataset_size,-1)
    forest = RandomForestClassifier(criterion = 'entropy', n_estimators = n_trees, random_state = 1, n_jobs = 4, verbose = True)
    forest.fit(X_train_TwoDim, Y_train)
    # Predicting the Test set results
    dataset_size = len(X_test)
    X_test_TwoDim = X_test.reshape(dataset_size,-1)
    y_pred = forest.predict(X_test_TwoDim)
    prob = forest.predict_proba(X_test_TwoDim)
    
    return forest, y_pred, prob
    
def createSVMmodel(X, Y, X_test):
    # Splitting the dataset into the Training set and Test set
    X = X/255
    X = np.array(X)
    Y= np.array(Y)
    X_train, Y_train = X, Y
    dataset_size = len(X_train)
    X_train_TwoDim = X_train.reshape(dataset_size,-1)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_TwoDim, Y_train,  
                                                          test_size=0.01, random_state=1111)
    
    # Fitting Kernel SVM to the Training set

    classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
    classifier.fit(X_train, Y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_valid)
    prob = classifier.predict_proba(X_valid)
    
    # Predicting the Test set results
    dataset_size = len(X_test)
    X_test_TwoDim = X_train.reshape(dataset_size,-1)
    y_pred_test = classifier.predict(X_test_TwoDim)
    prob_test = classifier.predict_proba(X_test_TwoDim)
    
    return classifier, y_pred, prob, y_pred_test, prob_test
    
        
    
