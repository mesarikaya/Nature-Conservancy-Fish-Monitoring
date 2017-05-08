# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:27:09 2017

@author: MustafaErgin
"""
import model_creation
import numpy as np
import pandas as pd
from keras.utils import np_utils
# set paths and caterogies
url = 'C:/Users/MustafaErgin/Desktop/Kaggle Competitions/The Nature Convervancy Fisheries Monitoring'
fish_types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
categories = [i+1 for i in range(len(fish_types))]
python_data_url = url + "/" + 'Python Data' + '/'

# Preprocess train and test datasets
x_train, y_train =  getImage(url = url, sub_folders = fish_types, type = 'train')
x_test =  getImage(url = url, sub_folders = None, type = 'test')

save_to_local(files = [x_train, y_train, x_test], file_names = ["X_train", "y_train", "x_test2"])
x_train = load_from_local("x_train", dest = python_data_url)
y_train = load_from_local("y_train", dest = python_data_url)
x_test = load_from_local("x_test", dest = python_data_url)

test_file_names = np.array([i[0] for i in x_test])
test_file_names = test_file_names.reshape(len(x_test), 1)

X_train = np.array(x_train, dtype=np.uint8)
X_train = X_train.astype('float32')

X_test = [i[1] for i in x_test]
X_test = np.array(X_test, dtype = np.uint8)
X_test = X_test.astype('float32')
X_test = X_test/255



Y_train = [fish_types.index(i) for i in y_train]
Y_train = np.array(Y_train, dtype=np.uint8)
Y_train = np_utils.to_categorical(Y_train, 8)



# Rescale and preprocess te images
datagen = ImageDataGenerator(
    featurewise_center=True, 
    featurewise_std_normalization=True,
    zca_whitening=True,
    rotation_range = 90,
    width_shift_range= 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True)

datagen.fit(X_train, seed=1)
save_to_local(files = [datagen], file_names = ["datagen"])
datagen = load_from_local("datagen", dest = python_data_url)
#m = datagen.flow(X_train, Y_train, batch_size=100, seed =1)
# Building the model
model_2 = createCnnModel(X_train, Y_train, datagen, input_epochs = 500)



model3 = createRFmodel(X_train, Y_train, X_test, datagen, n_trees = 500)


model4 = createSVMmodel(x_train, y_train, X_test)

# Predicting the Test set results
# Rescale and preprocess te images



y_pred = model_2.predict(X_test)
rf_y_pred = model3[2][0]

# Creating Submission fimle
submission = np.hstack((test_file_names, y_pred))
names = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
submission = pd.DataFrame(submission, columns=names)

# Save to excel csv
submission.to_csv(url+'/Submission/'+'submissionfinal.csv', index = False, encoding='utf-8')

# Creating Submission file
rf_submission = np.hstack((test_file_names, rf_y_pred))
names = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
rf_submission = pd.DataFrame(rf_submission, columns=names)

# Save to excel csv
rf_submission.to_csv(url+'/Submission/'+'rfsubmission.csv', index = False, encoding='utf-8')

# Save Keras cnn model
from keras.models import load_model
import h5py

model_2.save(python_data_url +'Fish_detection_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')


# y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_t, y_pred)