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
from tensorflow.keras.layers import Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications.resnet50 import preprocess_input
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


mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


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


def return_count(x):
    name_wel = []
    for i in sorted(x, key = natural_keys):
        name_wel.append(i.split('_')[0])

    z = sorted(list(set(name_wel)))
    r = list(range(len(z)))

    num = []
    for iz in range(len(z)):
        count = 0
        for i in sorted(x, key=natural_keys):
            if z[iz] in i:
                count += 1
        num.append(count)
    return list(zip(z, r, num))


# In[ ]:


def loadImages_LSTM(path_data,len_t_points):
    

    feat_list = []


    for filename in sorted(glob.glob(path_data), key=natural_keys): 
        feat_list.append(np.load(filename))

    x_orig = np.reshape(feat_list, (len(feat_list),len_t_points, 64))

    return x_orig 


# In[ ]:


def make_labels(data_set):
    fe = return_count(data_set)
    leb = creat_label(fe)
    y = np.array(list(leb))
    return y
    


# In[ ]:


def make_labels_LSTM(data_set):
    fe = return_count_LSTM(data_set)
    leb = creat_label(fe)
    y = np.array(list(leb))
    return y
    


# In[ ]:


def return_count_LSTM(x):
    name_wel = []
    for _,_,i in os.walk(x):
        for f in i:
            name_wel.append(f.split('_')[2])

    z = sorted(list(set(name_wel)), key=natural_keys)
    r = list(range(len(z)))

    num = []
    for iz in range(len(z)):
        count = 0
        for i in sorted(name_wel, key=natural_keys):
            if z[iz] in i:
                count += 1
        num.append(count)
    return list(zip(z, r, num))


# In[ ]:


def creat_label(y):
    labels = []
    for ix, _ in enumerate(y):
        
        if y[ix][0] == 'adr':
        
            labels.append([[y[ix][0],0]] * y[ix][2])
        
        if y[ix][0] == 'hrh':
            
            labels.append([[y[ix][0],1]] * y[ix][2])
        
            
        if y[ix][0] == 'dmso':
            
            labels.append([[y[ix][0],2]] * y[ix][2])
    
    ler = [i for sub in labels for i in sub ]
    
    _, lab= zip(*ler)

    
    return lab


# In[ ]:


def time_step_acc(tes_data, x):

    results = []            

    x_test = loadImages(tes_data)
    y_test = make_labels(tes_data)
    
    y_test_1 = keras.utils.to_categorical(y_test,num_classes=3)
    
    #x_test = resize(x_test)
    x_test = preprocess_input(x_test)

    scores = x.evaluate(x_test, y_test_1, verbose = 1)
    results.append(scores[1]*100)
    
    test_preds = m4.predict(x_test)

    preds_df = pd.DataFrame(test_preds)
    predicted_labels = preds_df.idxmax(axis=1)
    draw_confusion_matrix(y_test, predicted_labels)

    return results


# In[ ]:


def cv_mean_acc(result_cv, string_well):
    
    l_drug = string_well*3

    acc_mean_cv = []

    for i in result_cv:
        acc_mean_cv.append(np.mean(i))
        
    cv_drug = list(zip(acc_mean_cv, l_drug))
    
    res = sorted(cv_drug, key = lambda x: x[1])
    a , b = zip(*res)
    
    a = list(a)
    
    s = list(np.array_split(a, len(string_well)))
    
    cv_score_acc = []
    
    for ix, i in enumerate(s):
        s1 = list(s[ix])
        
        cv_score_acc.append(np.mean(s1))
        
    return list(zip(cv_score_acc, string_well))


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


tot_results_accuracy = []

results_lstm = []


# In[ ]:


time_points = list(map(str, range(0,34)))

new_time = []
for i in time_points:
    r = '_' + i + '.'
    new_time.append(r)


# In[ ]:


rand = list(np.random.randint(1,1000,3))


# In[ ]:


for num_ix, rand_num in enumerate(rand):
    for index_t_well, _ in tqdm.tqdm(enumerate(tot_well)):



        path_test = '/scratch-shared/victor/IMG_A549/{}_tile/'.format(a)

        # NAME OF THE WELLS CORRESPONDING TO THE DRUG THAT YOU WANT IN THE TEST SET 

        wells_drug = [tot_well[index_t_well][0]] 

        test = []

        for _,_, filenames in os.walk(path_test):

            for filename in sorted(filenames, key = natural_keys):

                for w in wells_drug:
                    for t in new_time:
                        if '{}'.format(w) in filename and '{}tiff'.format(t) in filename:
                            test.append(filename)

        groups_list = ['{}'.format(a), '{}'.format(b), '{}'.format(c)]

        fileds_of_view = ['1','2','3','4','5']

        field_train, field_val = train_test_split(fileds_of_view, test_size=0.4, random_state=rand_num)


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
                                                    class_mode="categorical", target_size=(256, 256), batch_size=64)

        val_generator=datagen.flow_from_dataframe(dataframe=df_val, directory = None , x_col="id", y_col="labels",color_mode="rgb", 
                                                    class_mode="categorical", target_size=(256, 256), batch_size=64)


        weights = class_weight.compute_class_weight('balanced', np.unique(train_leb), train_leb)
        
        weights = dict(enumerate(weights))
        


        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=3)

        pretrained_model = ResNet50(weights='imagenet',include_top=False, input_shape=(256, 256, 3))
        base_model = Model(inputs=pretrained_model.input, outputs=pretrained_model.get_layer('conv2_block3_out').output)
        
        print('Model_loaded')

        m4 = Sequential()
        m4.add(base_model)


        m4.add(BatchNormalization())
        m4.add(GlobalAveragePooling2D())
        m4.add(Dense(128, activation='tanh'))
        m4.add(BatchNormalization())
        m4.add(Dense(64, activation='tanh'))
        m4.add(BatchNormalization())
        m4.add(Activation('tanh'))
        m4.add(Dense(3,activation='softmax'))


        base_model.trainable = False

        opt = keras.optimizers.RMSprop(lr=1e-3)

        m4.compile(loss= keras.losses.categorical_crossentropy, optimizer=opt, metrics = ['accuracy'])


        epochs = 50
        
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

        m4_h = m4.fit(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, validation_data = val_generator, 
                      validation_steps = STEP_SIZE_VALID , callbacks = [es],
                      class_weight = weights, epochs=epochs, verbose = 1)


        base_model.trainable = True

        opt = keras.optimizers.Adam(lr=1e-4)

        m4.compile(loss= keras.losses.categorical_crossentropy, optimizer=opt, metrics = ['accuracy'])

        epochs = 300

        m4_h = m4.fit(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, validation_data = val_generator, 
                      validation_steps = STEP_SIZE_VALID , callbacks = [es],
                      class_weight = weights, epochs=epochs, verbose = 1)

        l = []
        for t in new_time:
            for i in test:
                if t in i:
                    l.append((i))


        grouped = {}
        for elem in l:
            key = elem.split('.tiff')[0].split('_')[5]
            grouped.setdefault(key, []).append(elem)
        grouped = grouped.values()

        test_data = list(grouped)

        r = []

        for ix ,_ in enumerate(test_data):
            r.append(time_step_acc(test_data[ix],m4))

        plt.plot(time_points,r)
        plt.savefig('/home/jovyan/IMG_CNN_FINAL/{}_accuracy.png'.format(string_well[index_t_well]))

        tot_results_accuracy.append(r)
        
        for i, layer in enumerate(m4.layers):
            layer._name = 'layer_' + str(i)



        lstm_model = Model(inputs=m4.input, outputs=m4.get_layer('layer_6').output)

        del m4
        K.clear_session()
        
        data_name = [train,test,validation]

        feat_name = ['train', 'test', 'validation']

        for index_name, _ in enumerate(data_name):

            path =  data_name[index_name]

            name_well = []

            for i in path:
                name_well.append(i.split('_id')[0])

            wells = list(set(name_well))
            wells

            for w in wells:

                time = []


                for filename in sorted(path, key = natural_keys):
                    if w in filename: #PAY ATTENTION ID THE IMAGE IS A TIFF OR PNG IMAGE #########
                        time.append(filename)

                data_id = {}
                n_id = []
                w_n = []

                for i in time:
                    t = i.split('_id_')[1].split('time_')[0]
                    f = i.split('_id_')[0].split('time_')[0]
                    n_id.append(t)
                    w_n.append(f)

                id_cell = set(n_id)


                for ix, i in enumerate(sorted(id_cell, key = natural_keys)):

                    id_name = []
                    dict_1 = {}

                    for t in time:
                        if 'id_{}'.format(i) in t:
                            id_name.append(t)

                    d = {'id':id_name}
                    data = pd.DataFrame(d)

                    dict_1[ix]=data 
                    data_id.update(dict_1) 


                len_id = [i for i, j in data_id.items()]

                for le in len_id:    


                    e = pd.DataFrame(data_id[le])

                    coords = e.values.tolist()
                    id_cells = []
                    for i in coords:
                        for j in i:
                            id_cells.append(j)
                            
                    

                    x_orig = loadImages(id_cells)
                    
                    x_orig = preprocess_input(x_orig)
                    output = lstm_model.predict(x_orig)
                    np.save('/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/features_{}/features_well_{}_id_{}.npy'.format(feat_name[index_name],w_n[0], le), output)
            print('Saved_feature_{}'.format(feat_name[index_name]))


        x_train_lstm = loadImages_LSTM(train_data, len(time_points))
        y_train_lstm = make_labels_LSTM(y_tra_path)

        x_test_lstm = loadImages_LSTM(tes_data, len(time_points))
        y_test_lstm = make_labels_LSTM(y_tes_path)

        x_val_lstm = loadImages_LSTM(val_data, len(time_points))
        y_val_lstm = make_labels_LSTM(y_val_path)

        weights_lstm = class_weight.compute_class_weight('balanced', np.unique(y_train_lstm),y_train_lstm)
        
        weights_lstm = dict(enumerate(weights_lstm))
        
        y_train_lstm = keras.utils.to_categorical(y_train_lstm, num_classes = 3)
        y_test_1_lstm = keras.utils.to_categorical(y_test_lstm, num_classes= 3)
        y_val_lstm = keras.utils.to_categorical(y_val_lstm, num_classes = 3)


        m = Sequential()
        m.add(LSTM(32, input_shape = (x_train_lstm.shape[1],x_train_lstm.shape[2])))
        m.add(Dropout(0.2))
        m.add(Dense(3, activation='softmax'))


        opt_lstm = keras.optimizers.Adam(lr=1e-4)

        m.compile(loss= keras.losses.categorical_crossentropy, optimizer=opt, metrics = ['accuracy'])

        epochs = 300

        m_h = m.fit(x_train_lstm,y_train_lstm,

                         callbacks = [es],

                        epochs=epochs,
                        validation_data = (x_val_lstm,y_val_lstm), 

                        class_weight = weights_lstm)


        scores_lstm = m.evaluate(x_test_lstm, y_test_1_lstm)
        results_lstm.append([scores_lstm[1]*100, string_well[index_t_well]])
        
        del m
        K.clear_session()

        # DELITE FILES IN FEATURE VECTOR FOLDERS

        folders = glob.glob('/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/*')

        for fo in folders:
            file = glob.glob(f'{fo}/*')
            for f in file:
                os.remove(f)


# In[ ]:


#### ACCURACY SCORE AVERAGE FOR CNN
cv_s = cv_mean_acc(tot_results_accuracy, string_well)
cv_s


# In[ ]:


cv_s_mean,_ = zip(*cv_s)

m_cv = np.mean(list(cv_s_mean))
m_cv    


# In[ ]:


# PLOT OF MEAN ACCURACY FOR EVERY TIME POINT CNN

l_drug = string_well*3

acc_plot = []

for i in tot_results_accuracy:
    acc_plot.append(i)

cv_plot = list(zip(acc_plot, l_drug))

res_plot = sorted(cv_plot, key = lambda x: x[1])

a , b = zip(*res_plot)
    
a = list(a)

s = list(np.array_split(a, len(string_well)))

cv_plot = []

for ix, i in enumerate(s):
    s1 = list(s[ix])
    
    cv_plot.append(np.average(s1, axis=0))


# In[ ]:


fig = plt.figure(figsize=(10, 5))
for i in cv_plot:
    
    plt.plot(time_points, i)
    plt.show
    plt.savefig('/home/jovyan/cv_score.png')


# In[ ]:


results_lstm = sorted(results_lstm, key=lambda x: x[1])
r_lstm , _ = zip(*results_lstm)
    
r_lstm = list(r_lstm)

re_lstm = list(np.array_split(r_lstm, len(string_well)))

cv_lstm = []

for ix, i in enumerate(re_lstm):
    r1 = list(re_lstm[ix])
    cv_lstm.append(np.mean(r1))


# In[ ]:


cv_lstm = list(zip(cv_lstm, string_well))
cv_lstm


# In[ ]:


cv_l_mean,_ = zip(*cv_lstm)

m_cv_l = np.mean(list(cv_l_mean))
m_cv_l 


# In[ ]:


# BEFORE RUNNING AGAIN

folders = glob.glob('/home/jovyan/DATA_MASTER_PROJECT/LSTM/FEAT_FOLDERS/*')

for fo in folders:
    file = glob.glob(f'{fo}/*')
    for f in file:
        os.remove(f)


# In[ ]:


cv_l_mean,_ = zip(*cv_lstm)

m_cv_l = np.mean(list(cv_l_mean))
m_cv_l 


# In[ ]:


sd_cv_l =  np.std(list(cv_l_mean))
sd_cv_l


# In[ ]:


me = np.mean(cv_plot, axis = 0)
me = me.flatten()


# In[ ]:


sd = np.std(cv_plot, axis = 0)
sd = sd.flatten()


# In[ ]:


plt.rc('ytick', labelsize=18)
ax = plt.figure(figsize=(13,8), facecolor='w').gca()
ax.plot(me)
ax.fill_between(time_points, me - sd, me + sd, alpha = 0.5)

plt.errorbar(31.5, m_cv_l, sd_cv_l, linestyle='None', marker='^', markersize=12)

plt.savefig('/home/jovyan/md_sd_score.png')


# In[ ]:


cnn_stat = [i for i in cv_s_mean]
lstm_stat = [i for i in cv_l_mean]

from scipy.stats import wilcoxon

stat, p = wilcoxon(cnn_stat, lstm_stat)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')


# In[ ]:





# In[ ]:




