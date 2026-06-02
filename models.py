
import argparse,sys, os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import LeavePGroupsOut
import random
import pickle
import torch.optim.lr_scheduler as lr_Scheduler


def getModels(model_num):

    if model_num == 1:
        extractor = Extractor1()
        classifier = classifier1()

    return extractor,classifier




class Extractor1(nn.Module):
    def __init__(self,dilation_fac=2):
        super(Extractor1, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=20,
                kernel_size=(40,1),
                stride=(1,1),
        )

        self.bn1 = nn.BatchNorm2d(20,
                        momentum=0.1,
                        affine=True,
                        eps=1e-5,
                    )


        self.conv2 = nn.Conv2d(
                in_channels=20,
                out_channels=80,
                kernel_size=(1,19),
                stride=(1,1),


        )
        self.bn2 = nn.BatchNorm2d(80,
                        momentum=0.1,
                        affine=True,
                        eps=1e-5,
                    )


        self.conv3 = nn.Conv2d(
                in_channels=80,
                out_channels=100,
                kernel_size=(5,1),
                stride=(1,1),
                dilation=dilation_fac,
        )
        self.bn3 = nn.BatchNorm2d(100,
                        momentum=0.1,
                        affine=True,
                        eps=1e-5,
                    )
        self.dr3 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
                in_channels=100,
                out_channels=160,
                kernel_size=(10,1),
                stride=(1,1)
        )
        self.bn4 = nn.BatchNorm2d(160,
                        momentum=0.1,
                        affine=True,
                        eps=1e-5,
                    )




    def forward(self,x):


        x=self.conv1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.elu(x)
        x=F.max_pool2d(x,kernel_size=(5,1), stride=(5,1))

        x=self.conv3(x)
        x=self.bn3(x)
        x=F.elu(x)
        x=F.max_pool2d(x,kernel_size=(5,1), stride=(5,1))
        x=self.dr3(x)

        x=self.conv4(x)


        x=self.bn4(x)
        x=F.elu(x)
        x=F.max_pool2d(x,kernel_size=(5,1), stride=(5,1))


        x = x.view(-1, 800)


        return x


class classifier1(nn.Module):
    def __init__(self):
        super(classifier1, self).__init__()

        self.fc1 = nn.Linear(800,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,2)
        self.bn2 = nn.BatchNorm1d(2)

        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self,x):


        x=F.dropout(x,0.5)
        x=self.fc1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.logsoftmax(x)   

        return x

