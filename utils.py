import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # suppress deprecation messages
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from alibi.explainers import CounterFactual
from alibi.explainers import CEM
from PIL import Image
import pandas as pd
import sys
def mnist_load():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)

    x_train = (x_train.astype('float32') / 255) - 0.5
    x_test = (x_test.astype('float32') / 255) - 0.5
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))
    print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)


    return x_train,y_train,x_test,y_test
def tabulr_train_load(train_path):
    min, ran = tabular_data_info(train_path)

    load_df=pd.read_csv(train_path)
    tmp=pd.DataFrame(load_df.iloc[0])
    tmp=tmp.T
    features, label_name = get_col_name(train_path, tmp)
    tr_data = pd.DataFrame(load_df, columns=features)
    tr_data = (np.array(tr_data)-min)/ran
    return tr_data,features
def get_col_name(data_path, x):

    if 'heloc' in data_path or'HELOC' in data_path:
        label_name = 'RiskPerformance'
    elif 'UCI_Credit' in data_path:
        label_name = 'default.payment.next.month'

    else:
        print('label name not vaild!')
        sys.exit(0)
    features = [c for c in x.columns if c != label_name]

    return features, label_name
def save_img(input,arg_path,file_path):
    input=np.array(input)
    fig=np.around((input+0.5)*255)
    fig=fig.astype(np.uint8).squeeze()
    pic=Image.fromarray(fig)
    pic.save("Results/MNIST/{}/{}".format(arg_path,file_path))

def save_csv(input,base_path,arg_path,file_name,features):
    df_in=pd.DataFrame(data=input,columns=features)

    path=base_path+'/'+arg_path+'/'+file_name+'.csv'

    df_in.to_csv(path)



def tabular_data_info(model_path):
    if 'heloc' in model_path or 'HELOC' in model_path:
        min = np.array([-8.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, \
                        2.0, 0.0, 0.0, 0.0, -8.0, 0.0, 0.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0])

        ran = np.array([797.0, 159.0, 254.0, 67.0, 14.0, 10.0, 100.0, 89.0, 9.0, 6.0, 83.0, \
                        16.0, 100.0, 32.0, 46.0, 46.0, 134.0, 173.0, 40.0, 21.0, 21.0, 108.0])

    elif 'UCI_Credit' in model_path:
        min = np.array([10000.0, 1.0, 0.0, 0.0, 21.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, \
                        -15308.0, -33350.0, -46127.0, -50616.0, -81334.0, -339603.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ran = np.array([790000, 1, 6, 3, 54, 10, 9, 10, 10, 10, 10, 637057, 777320, \
                        605839, 667452, 612006, 853401, 493358, 1227082, 380478, \
                        400046, 388071, 403500])

    return min,ran