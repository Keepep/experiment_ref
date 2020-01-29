import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # suppress deprecation messages
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
def main():
    x_train, y_train, x_test, y_test = mnist_load()
    train=False

    if train==True:
        cnn = cnn_model()
        cnn.summary()
        cnn.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1)
        cnn.save('saved_classifer/mnist_cnn.h5', save_format='h5')
        ae = ae_model()
        ae.summary()
        ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
        ae.save('saved_classifer/mnist_ae.h5', save_format='h5')

    cnn = load_model('mnist_pytorch.h5')
    ae = load_model('mnist_ae.h5')
    for i in range(1000):
        idx = i
        X = x_test[idx].reshape((1,) + x_test[idx].shape)
        plt.imshow(X.reshape(28, 28))
        #plt.show()

        pred_class=cnn.predict(X).argmax()
        pred_prob=cnn.predict(X).max()
        print(pred_prob)
        print(pred_class)

        mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
        shape = (1,) + x_train.shape[1:]  # instance shape
        kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
                    # class predicted by the original instance and the max probability on the other classes
                    # in order for the first loss term to be minimized
        beta = .1  # weight of the L1 loss term
        gamma = 100  # weight of the optional auto-encoder loss term
        c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or
                      # the same class (PP) for the perturbed instance compared to the original instance to be explained
        c_steps = 10  # nb of updates for c
        max_iterations = 1000  # nb of iterations per value of c
        feature_range = (x_train.min(),x_train.max())  # feature range for the perturbed instance
        clip = (-1000.,1000.)  # gradient clipping
        lr = 1e-2  # initial learning rate
        no_info_val = -1. # a value, float or feature-wise, which can be seen as containing no info to make a prediction
                          # perturbations towards this value means removing features, and away means adding features
                          # for our MNIST images, the background (-0.5) is the least informative,
                          # so positive/negative perturbations imply adding/removing features

        cem = CEM(cnn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
                  gamma=gamma, ae_model=ae, max_iterations=max_iterations,
                  c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

        explanation = cem.explain(X)

        try:
            print('Pertinent negative prediction: {}'.format(explanation[mode + '_pred']))
        except:
            continue
        plt.imshow(explanation[mode].reshape(28, 28))
        #plt.show()
        arg_save_dir = "ID{}".format(idx)
        org_save_file="Org_class{}.png".format(pred_class)
        per_save_file="Per_class{}.png".format(explanation[mode + '_pred'])
        delta_save_file="Delta.png"
        os.system("mkdir -p Results/MNIST/{}".format(arg_save_dir))

        delta_img=np.absolute(X-explanation[mode])-0.5


        save_img(X,arg_save_dir,org_save_file)
        save_img(explanation[mode],arg_save_dir,per_save_file)
        save_img(delta_img,arg_save_dir,delta_save_file)

        output_org=cnn.predict(X).max()
        output_comp=cnn.predict(explanation[mode]).max()

        result = 'output_org: ' + str(output_org) + '\n' + \
                'class_org: '+str(explanation['X_pred'])+'\n'+\
                 'output_comp: ' + str(output_comp)+'\n'+ \
                 'class_comp: ' + str(explanation[mode+'_pred'])

        f = open("Results/MNIST/{}".format(arg_save_dir)+'/' + 'result.txt', 'w')
        f.write(result)
        f.close()
if __name__ == '__main__':
    main()
