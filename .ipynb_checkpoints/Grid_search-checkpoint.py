#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image 
import seaborn as sns
import os
import re
import glob
import cv2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import tqdm
from numpy import loadtxt
from os import *
from sklearn.utils import class_weight


# In[ ]:


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# In[ ]:


def draw_confusion_matrix(true,preds):
    conf_matx = confusion_matrix(true, preds)
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap=plt.cm.Blues) #'viridis'
    #plt.savefig('/home/jovyan/conf_matrix.png')
    plt.show()
    
    return conf_matx


# In[ ]:


def plot_history(model_history, model_name):
    fig = plt.figure(figsize=(15,5), facecolor='w')
    ax = fig.add_subplot(121)
    ax.plot(model_history.history['loss'])
    ax.plot(model_history.history['val_loss'])
    ax.set(title=model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
    ax.legend(['Train', 'Val'], loc='upper left')
    ax = fig.add_subplot(122)
    ax.plot(model_history.history['accuracy'])
    ax.plot(model_history.history['val_accuracy'])
    ax.set(title=model_name + ': Model Accuracy; test='+ str(np.round(model_history.history['val_accuracy'][-1], 3)),
           ylabel='Accuracy', xlabel='Epoch')
    ax.legend(['Train', 'Val'], loc='upper left')
    #plt.savefig('/home/jovyan/curve.png')
    plt.show()
    
    return fig


# In[ ]:


def loadImages(path_data):
    
    p = '/home/jovyan/DATA_MASTER_PROJECT/IMG_A549_high_con/'
    
    
    
    pa_adr = p + 'ADR_tile/'
    
    pa_control = p + 'CONTROL_cropped/'
    
    pa_hrh = p + 'HRH_tile/'
    
    pa_dmso = p + 'DMSO_tile/'
    
    image_list = []
    
    
       


    for filename in sorted(path_data, key=natural_keys): 
        
        if 'adr' in filename:
            
            im=cv2.imread(pa_adr + filename,1)

            imarray = np.array(im)
            

            image_list.append(imarray)
        
            
        if 'hrh' in filename:
            
            im=cv2.imread(pa_hrh + filename,1)

            imarray = np.array(im)
            

            image_list.append(imarray)
            
        if 'dmso' in filename:
            
            im=cv2.imread(pa_dmso + filename,1)

            imarray = np.array(im)
            

            image_list.append(imarray)



    x_orig = np.reshape(image_list, (len(image_list), 256, 256, 3))

    return x_orig


# In[ ]:


def Images_path(path_data):
    
    p = '/scratch-shared/victor/IMG_A549/'
    
    
    
    pa_adr = p + 'ADR_tile/'
    pa_hrh = p + 'HRH_tile/'
    pa_dmso = p + 'DMSO_tile/'
    
    image_list = []
    
    
       


    for filename in path_data: 
        
        if 'adr' in filename:
            

            image_list.append(pa_adr + filename)
        
            
        if 'hrh' in filename:

            image_list.append(pa_hrh + filename)
            
        if 'dmso' in filename:
            

            image_list.append(pa_dmso + filename)

    return image_list 


# In[ ]:


def label(y):
    lab = []
    for i in y:
        if 'adr' in i:
            lab.append(0)
        if 'hrh' in i:
            lab.append(1)
        if 'dmso' in i:
            lab.append(2)
    return lab


# In[ ]:


# DATA FOR LSTM PART

p_feat = '/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/'
train_data = p_feat + 'features_train/*.npy'
val_data = p_feat + 'features_validation/*.npy'
tes_data= p_feat + 'features_test/*.npy'

y_tra_path = '/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/features_train/'
y_tes_path = '/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/features_test/'
y_val_path = '/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/features_validation/'


# In[ ]:


met = ['F10']
mid = ['D5']
oxy = ['F6']




cycl = ['C4']
dime =  ['F7']
cypr  = ['G9']


# In[ ]:


tot_well_adr = [met,mid,oxy]

tot_well_hrh = [cycl, dime, cypr]

string_well_adr = ['met', 'mid', 'oxy']

string_well_hrh = ['cycl', 'dime', 'cypr']


# In[ ]:


tot_well = []
string_well = [] 


# In[ ]:


a = 'ADR' # FOR TEST SET
b = 'HRH' # FOR REST
c = 'DMSO'

if a == 'HRH':
    tot_well = tot_well_hrh
    string_well = string_well_hrh
    
if a == 'ADR':
    tot_well = tot_well_adr
    string_well = string_well_adr
    


# In[ ]:


time_points = list(map(str, range(0,34)))

new_time = []
for i in time_points:
    r = '_' + i + '.'
    new_time.append(r)


# In[ ]:


path_test = '/scratch-shared/victor/IMG_A549/{}_tile/'.format(a)

# NAME OF THE WELLS CORRESPONDING TO THE DRUG THAT YOU WANT IN THE TEST SET 

wells_drug = [tot_well[0][0]] 

test = []

for _,_, filenames in os.walk(path_test):

    for filename in sorted(filenames, key = natural_keys):

        for w in wells_drug:
            for t in new_time:
                if '{}'.format(w) in filename and '{}tiff'.format(t) in filename:
                    test.append(filename)

groups_list = ['{}'.format(a), '{}'.format(b), '{}'.format(c)]

fileds_of_view = ['1','2','3','4','5']

field_train, field_val = train_test_split(fileds_of_view, test_size=0.4, random_state=124)


train = []

validation = []

group_compounds = []

all_wells = tot_well_adr + tot_well_hrh
all_wells.remove(wells_drug)
allw = [j for i in all_wells for j in i]

for group in tqdm.tqdm(groups_list):

    pa = '/scratch-shared/victor/IMG_A549/{}_tile/'.format(group)

    for _,_, filenames in os.walk(pa):

        for filename in sorted(filenames, key = natural_keys):

            for t in new_time:

                for al in allw:

                    if '_{}-'.format(al) in filename  and '{}tiff'.format(t) in filename:

                        group_compounds.append(filename)




pp = '/scratch-shared/victor/IMG_A549/DMSO_tile/'

dm = []
for _,_, filenames in os.walk(pp): 

    for f in filenames:
        dm.append(f)


group_compounds = group_compounds + dm

for i in group_compounds:

    for f in field_train:
        if '-{}_'.format(f) in i:
            train.append(i)


    for v in field_val:
        if '-{}_'.format(v) in i:
            validation.append(i)


# In[ ]:


tot_results_accuracy = []
op = [Adam, RMSprop]
n_op = ['Adma', 'RMSP']
batch_size = [64,128]
act = ['relu', 'tanh' ]
comb = []
for a in tqdm.tqdm(act):
    for b in batch_size:
        
        for o, n in zip(op, n_op):

            comb.append(list((a,b, n)))
            
            

            train_leb = label(train)
            val_leb = label(validation)
            test_leb = label(test)
            train_cnn = Images_path(train)
            validation_cnn = Images_path(validation)


            data_train = {'id': train_cnn, 'labels':train_leb}
            data_val = {'id': validation_cnn, 'labels':val_leb}

            df_train = pd.DataFrame(data_train, columns = ['id', 'labels'])
            df_val = pd.DataFrame(data_val, columns = ['id', 'labels'])

            df_train['labels'] = df_train['labels'].astype(str)
            df_val['labels'] = df_val['labels'].astype(str)

            datagen=ImageDataGenerator(preprocessing_function = preprocess_input)

            train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory = None , x_col="id", y_col="labels",color_mode="rgb", 
                                                        class_mode="categorical", target_size=(256, 256), batch_size=b)

            val_generator=datagen.flow_from_dataframe(dataframe=df_val, directory = None , x_col="id", y_col="labels",color_mode="rgb", 
                                                        class_mode="categorical", target_size=(256, 256), batch_size=b)


            weights = class_weight.compute_class_weight('balanced', np.unique(train_leb), train_leb)

            weights = dict(enumerate(weights))



            es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=3)

            pretrained_model = VGG16(weights='imagenet',include_top=False, input_shape=(256, 256, 3))

            base_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.get_layer('block3_pool').output)

            print('Model_loaded')

            m4 = Sequential()
            m4.add(base_model)


            m4.add(BatchNormalization())
            m4.add(GlobalAveragePooling2D())
            m4.add(Dense(128, activation=a))
            m4.add(BatchNormalization())
            m4.add(Dense(64, activation=a))
            m4.add(BatchNormalization())
            m4.add(Activation(a))
            m4.add(Dense(3,activation='softmax'))


            base_model.trainable = False

            opt = o(lr=1e-3)

            m4.compile(loss= keras.losses.categorical_crossentropy, optimizer=opt, metrics = ['accuracy'])



            epochs = 50

            STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
            STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

            m4_h = m4.fit(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, validation_data = val_generator, 
                          validation_steps = STEP_SIZE_VALID , callbacks = [es],
                          class_weight = weights, epochs=epochs, verbose = 1)


        

            x_test = loadImages(test)
            y_test = label(test)

            y_test_1 = keras.utils.to_categorical(y_test,num_classes=3)

            x_test = preprocess_input(x_test)

            scores = m4.evaluate(x_test, y_test_1, verbose = 1)

            tot_results_accuracy.append((scores[1]*100))


            del m4
            K.clear_session()

            



# In[ ]:


done = list(zip(tot_results_accuracy, comb))
sorted(done, reverse = True)


# In[ ]:




