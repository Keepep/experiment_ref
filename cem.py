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
from utils import *
from cnn_model import cnn_model
from ae_model import ae_model
from alibi.explainers import CounterFactual

def main():
    x_train, y_train, x_test, y_test = mnist_load()
    train=False

    if train==True:
        cnn = cnn_model()
        cnn.summary()
        cnn.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)
        cnn.save('saved_classifier/mnist_cnn.h5', save_format='h5')
        ae = ae_model()
        ae.summary()
        ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
        ae.save('saved_classifier/mnist_ae.h5', save_format='h5')

    cnn = load_model('mnist_cnn.h5')
    ae = load_model('mnist_ae.h5')
    for i in range(0,1000):
        idx = i
        X = x_test[idx].reshape((1,) + x_test[idx].shape)
        plt.imshow(X.reshape(28, 28));
        #plt.show()

        pred_class=cnn.predict(X).argmax()
        pred_prob=cnn.predict(X).max()

        shape = (1,) + x_train.shape[1:]
        target_proba = 1.0
        tol = 0.01  # want counterfactuals with p(class)>0.99
        target_class = 'other'  # any class other than 7 will do
        max_iter = 1000
        lam_init = 1e-1
        max_lam_steps = 10
        learning_rate_init = 0.1
        feature_range = (x_train.min(), x_train.max())

        cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                            target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                            max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                            feature_range=feature_range)



        explanation = cf.explain(X)

        try:
            print('Pertinent negative prediction: {}'.format(explanation['cf']['class']))
        except:
            continue
        plt.imshow(explanation['cf']['X'].reshape(28, 28))
        #plt.show()
        arg_save_dir = "ID{}".format(idx)
        org_save_file="Org_class{}.png".format(pred_class)
        per_save_file="Per_class{}.png".format(explanation['cf']['class'])
        delta_save_file="Delta.png"
        os.system("mkdir -p Results/{}".format(arg_save_dir))

        delta_img=np.absolute(X-explanation['cf']['X'])-0.5


        save_img(X,arg_save_dir,org_save_file)
        save_img(explanation['cf']['X'],arg_save_dir,per_save_file)
        save_img(delta_img,arg_save_dir,delta_save_file)

if __name__ == '__main__':
    main()
