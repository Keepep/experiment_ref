import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,  Input
from keras.utils import np_utils
from keras.models import load_model
from pytorch2keras import pytorch_to_keras
from torch.autograd import Variable
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv2d_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2d_2=nn.Conv2d(32, 32, 3, 1)

        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, 1)


        self.dense_1 = nn.Linear(1024, 200)
        self.dense_2 = nn.Linear(200, 200)
        self.dense_3 = nn.Linear(200, 10)


    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(x)
        x = self.conv2d_2(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2d_3(x)
        x = F.relu(x)
        x = self.conv2d_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dense_3(x)

        #x = F.log_softmax(x, dim=1)
        return x

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda
if __name__ == '__main__':
    model_path='mnist_cnn_pytorch.pt'
    if torch.typename(torch.load(model_path)) == 'OrderedDict':
        model = Net()
        model.load_state_dict(torch.load(model_path))

    else:
        model = torch.load(model_path)

    model.cpu()
    input_np = np.random.uniform(0, 1, (1,1,28, 28))
    print(np.shape(input_np))
    input_var = Variable(torch.FloatTensor(input_np))
    mnist_model=pytorch_to_keras(model,input_var,verbose=True)

    ex_path = 'examples/test0_0.png'
    org_data = Image.open(ex_path).convert('L')

    trans = transforms.Compose([ \
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    org_data_tensor = trans(org_data).float()
    org_data_numpy = org_data_tensor.cpu().numpy()
    org_data_tensor = torch.reshape(org_data_tensor, ( 1,1,28, 28))  # size=[1,1,28,28] (input size)
    org_data_numpy_reshape = org_data_tensor.cpu().numpy()


    output=mnist_model.predict(org_data_numpy_reshape)
    print(output)

    mnist_model.save('mnist_pytorch.h5')
    # mnistModel=MNISTModel()
    #
    # pth2keras(model,mnistModel.model)
    # mnistModel.model.save_weights( 'mnist_pytorch.h5')
