# %% codecell
import torch
import pickle
from torch.nn import ReLU
from matplotlib.pyplot import imshow
import fooof
from scipy.signal import welch
import argparse,sys, os
import copy
import time
import h5py
import numpy as np
import pandas as pd
import tables
import matplotlib.pyplot as plt
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
from models import getModels
from sklearn.manifold import TSNE
from matplotlib import cm
import seaborn as sns
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.fftpack import fft
import mne
from mne.viz import plot_topomap
import math
from PIL import Image
import io
from torch.autograd import Variable
import mne
from statsmodels.stats.multitest import fdrcorrection
import scipy
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
# %% codecell
if True:
    elec_names={
    0:'Fz',
    1:'Cz',
    2:'Pz',
    3:'C3',
    4:'C4',
    5:'T3',
    6:'T4',
    7:'Fp1',
    8:'Fp2',
    9:'O1',
    10:'O2',
    11:'F3',
    12:'F4',
    13:'P3',
    14:'P4',
    15:'FC1',
    16:'FC2',
    17:'CP1',
    18:'CP2'}
    CHANNEL_NAMES=['Fz', 'Cz', 'Pz', 'C3', 'C4', 'T3', 'T4', 'Fp1', 'Fp2', 'O1','O2', 'F3', 'F4', 'P3', 'P4', 'FC1', 'FC2', 'CP1', 'CP2']
    SENSORS_POS=np.array([
    [0.714,0,0.7],
    [6.12e-17,0,1],
    [-0.714,-8.74e-17,0.7],
    [4.55e-17,0.744,0.668],
    [4.55e-17,-0.744,0.668],
    [6.09e-17,0.995,-0.103],
    [6.09e-17,-0.995,-0.103],
    [0.95,0.309,-0.0471],
    [0.95,-0.309,-0.0471],
    [-0.95,0.309,-0.0471],
    [-0.95,-0.309,-0.0471],
    [0.677,0.568,0.468],
    [0.676,-0.567,0.471],
    [-0.677,0.568,0.468],
    [-0.676,-0.567,0.471],
    [0.381,0.381,0.843],
    [0.381,-0.381,0.843],
    [-0.381,0.381,0.843],
    [-0.381,-0.381,0.843]])[:,:2]
    t=SENSORS_POS
    t1=np.array(t[:,0])
    t[:,0]=t[:,1]
    t[:,1]=t1
    t[:,0] *= -1
    SENSORS_POS=t

# %% codecell
caps_sleep_stage='REM'
sleep_stage='rem'
only_good_windows =True
all_data= False
window_type = 1



results_dir = 'results/'+caps_sleep_stage+'-new3/'
allfiles=os.listdir(results_dir)
accuracy_arr=np.zeros(18)
for f in allfiles:
    if f[-3:]=='png':
        accuracy_arr[int(f.split('_')[2])]= int(f.split(':')[1].split('.')[1])

sleep_dir = 'visresults/'+sleep_stage+'/'
allfiles=os.listdir(sleep_dir)
valinum_array=[]
for f in allfiles:
    try:
        valinum_array.append(int(f.split('-')[1].split('d')[0]))
    except:
        a=1
valinum_array=list(set(valinum_array))

for vset in valinum_array:
    print(vset,accuracy_arr[vset])


# %% codecell
valinum_array1=list(valinum_array)
for vset in valinum_array:
    if(accuracy_arr[vset]<750):
        valinum_array1.remove(vset)
        print(vset)
valinum_array=valinum_array1
for vset in valinum_array:
    print(vset,accuracy_arr[vset])

# %% codecell
if only_good_windows==False:
    if all_data==False:
        allfinalarrays1 = []
        allfinalarrays2 = []
        num_samples = np.zeros(19)
        for vset in valinum_array:
            print(vset)
            log = False
            fooofsub = True

            program_type = 1
            if True:

                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)


                if program_type==1:
                    electrodeimp = []
                    for i in range(19):
                        electrodeimp.append(np.sum(impset1[i]))
                    electrodeimp = np.array(electrodeimp)
                    electrodeimp=electrodeimp/electrodeimp.max()



                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))


                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))

                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix
            allfinalarrays1.append(final_result)

            program_type = 2
            if True:

                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)


                if program_type==1:
                    electrodeimp = []
                    for i in range(19):
                        electrodeimp.append(np.sum(impset1[i]))
                    electrodeimp = np.array(electrodeimp)
                    electrodeimp=electrodeimp/electrodeimp.max()



                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))


                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))



                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix

            allfinalarrays2.append(final_result)

        print("arr1",len(allfinalarrays1),"arr2",len(allfinalarrays2))
# %% codecell
if only_good_windows==False:
    if all_data==True:
        allfinalarrays1 = []
        allfinalarrays2 = []

        window_type=0
        num_samples = np.zeros(19)
        for vset in valinum_array:

            log = False
            fooofsub = True

            program_type = 1
            if True:


                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)






                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))



                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))


                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix
            allfinalarrays1.append(final_result)

            program_type = 2
            if True:

                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)


                if program_type==1:
                    electrodeimp = []
                    for i in range(19):
                        electrodeimp.append(np.sum(impset1[i]))
                    electrodeimp = np.array(electrodeimp)
                    electrodeimp=electrodeimp/electrodeimp.max()



                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))



                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))


                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix

            allfinalarrays2.append(final_result)

        window_type=1
        num_samples = np.zeros(19)
        for vset in valinum_array:

            log = False
            fooofsub = True

            program_type = 1
            if True:

                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)


                if program_type==1:
                    electrodeimp = []
                    for i in range(19):
                        electrodeimp.append(np.sum(impset1[i]))
                    electrodeimp = np.array(electrodeimp)
                    electrodeimp=electrodeimp/electrodeimp.max()



                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))


                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))


                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix
            allfinalarrays1.append(final_result)

            program_type = 2
            if True:

                inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+ 'win-type-' + str(window_type)+'.data'
                with open(inputfilenamedata, 'rb') as filehandle:
                    # read the data as binary data stream
                    mainset1 = pickle.load(filehandle)
                with open(inputfilenameimp, 'rb') as filehandle:
                    # read the data as binary data stream
                    impset1 = pickle.load(filehandle)


                if program_type==1:
                    electrodeimp = []
                    for i in range(19):
                        electrodeimp.append(np.sum(impset1[i]))
                    electrodeimp = np.array(electrodeimp)
                    electrodeimp=electrodeimp/electrodeimp.max()



                Fs = 200
                window = mainset1[0].shape[1]
                random_seed=2
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)

                ffttype = 0#0-scipy, 1- welch
                endfoooffreq = 42
                freq_bands = [0,4,8,12,16,32,42]
                freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

                final_freq_matrix=[]
                for elec in range(19):
                    start = elec
                    end = start+1


                    alltimeset = np.empty((0,window))
                    for i in range(start,end):
                        alltimeset = np.concatenate((alltimeset,mainset1[i]))
                    allimpset = np.empty((0,window))
                    for i in range(start,end):
                        allimpset = np.concatenate((allimpset,impset1[i]))


                    if ffttype == 0:
                        Fs = 200
                        n_points = window
                        frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                    if ffttype == 1:
                        frequencies = newfr

                    def getfft(data):
                        if ffttype==0:
                            fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                        if ffttype==1:
                            fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                        return fftdata

                    ###########
                    allfft=[]
                    for i in range(alltimeset.shape[0]):
                        tt = getfft(alltimeset[i])
                        #timp = allimpset[i].sum()
                        allfft.append(tt)
                    if(len(allfft)!=0):
                        allfft=np.stack( allfft, axis=0 )
                    else:
                        allfft=np.empty((0,41))


                    final_freq_matrix.append(allfft)




                #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

                final_result = final_freq_matrix

            allfinalarrays2.append(final_result)

        window_type=3

        print("arr1",len(allfinalarrays1),"arr2",len(allfinalarrays2))

# %% codecell
if only_good_windows==True:
    program_type = 1

    allfinalarrays1 = []
    allfinalarrays2 = []
    num_samples = np.zeros(19)
    for vset in valinum_array:
        print(vset)
        log = False
        fooofsub = True

        window_type = 0
        if True:

            inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
            inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
            with open(inputfilenamedata, 'rb') as filehandle:
                # read the data as binary data stream
                mainset1 = pickle.load(filehandle)
            with open(inputfilenameimp, 'rb') as filehandle:
                # read the data as binary data stream
                impset1 = pickle.load(filehandle)


            if program_type==1:
                electrodeimp = []
                for i in range(19):
                    electrodeimp.append(np.sum(impset1[i]))
                electrodeimp = np.array(electrodeimp)
                electrodeimp=electrodeimp/electrodeimp.max()



            Fs = 200
            window = mainset1[0].shape[1]
            random_seed=2
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

            ffttype = 0#0-scipy, 1- welch
            endfoooffreq = 42
            freq_bands = [0,4,8,12,16,32,42]
            freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

            final_freq_matrix=[]
            for elec in range(19):
                start = elec
                end = start+1


                alltimeset = np.empty((0,window))
                for i in range(start,end):
                    alltimeset = np.concatenate((alltimeset,mainset1[i]))
                allimpset = np.empty((0,window))
                for i in range(start,end):
                    allimpset = np.concatenate((allimpset,impset1[i]))


                if ffttype == 0:
                    Fs = 200
                    n_points = window
                    frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                if ffttype == 1:
                    frequencies = newfr

                def getfft(data):
                    if ffttype==0:
                        fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                    if ffttype==1:
                        fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                    return fftdata

                ###########
                allfft=[]
                for i in range(alltimeset.shape[0]):
                    tt = getfft(alltimeset[i])
                    #timp = allimpset[i].sum()
                    allfft.append(tt)
                if(len(allfft)!=0):
                    allfft=np.stack( allfft, axis=0 )
                else:
                    allfft=np.empty((0,41))

                final_freq_matrix.append(allfft)




            #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

            final_result = final_freq_matrix
        allfinalarrays1.append(final_result)

        window_type = 1
        if True:

            inputfilenamedata =sleep_dir+ 'set-'+str(vset)+'data-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
            inputfilenameimp = sleep_dir+'set-'+str(vset)+'imp-pro-'+str(program_type)+'win-type-' + str(window_type)+'.data'
            with open(inputfilenamedata, 'rb') as filehandle:
                # read the data as binary data stream
                mainset1 = pickle.load(filehandle)
            with open(inputfilenameimp, 'rb') as filehandle:
                # read the data as binary data stream
                impset1 = pickle.load(filehandle)


            if program_type==1:
                electrodeimp = []
                for i in range(19):
                    electrodeimp.append(np.sum(impset1[i]))
                electrodeimp = np.array(electrodeimp)
                electrodeimp=electrodeimp/electrodeimp.max()



            Fs = 200
            window = mainset1[0].shape[1]
            random_seed=2
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

            ffttype = 0#0-scipy, 1- welch
            endfoooffreq = 42
            freq_bands = [0,4,8,12,16,32,42]
            freq_band_names=["delta","theta","alpha","sigma","beta","gamma"]

            final_freq_matrix=[]
            for elec in range(19):
                start = elec
                end = start+1


                alltimeset = np.empty((0,window))
                for i in range(start,end):
                    alltimeset = np.concatenate((alltimeset,mainset1[i]))
                allimpset = np.empty((0,window))
                for i in range(start,end):
                    allimpset = np.concatenate((allimpset,impset1[i]))


                if ffttype == 0:
                    Fs = 200
                    n_points = window
                    frequencies = (Fs/2)*np.linspace(0,1,1+int(n_points/2))
                if ffttype == 1:
                    frequencies = newfr

                def getfft(data):
                    if ffttype==0:
                        fftdata=(2/n_points)*abs(fft(data)[:len(frequencies)])
                    if ffttype==1:
                        fr, fftdata = scipy.signal.welch(data,fs=200, window='hamming',nperseg=window)
                    return fftdata

                ###########
                allfft=[]
                for i in range(alltimeset.shape[0]):
                    tt = getfft(alltimeset[i])
                    #timp = allimpset[i].sum()
                    allfft.append(tt)
                if(len(allfft)!=0):
                    allfft=np.stack( allfft, axis=0 )
                else:
                    allfft=np.empty((0,41))



                final_freq_matrix.append(allfft)




            #final_freq_matrix = np.stack(final_freq_matrix,axis=0)

            final_result = final_freq_matrix

        allfinalarrays2.append(final_result)

    print("arr1",len(allfinalarrays1),"arr2",len(allfinalarrays2))

    window_type = 5
# %% codecell



# %% codecell


elec_wise_results1=[]
for elec in range(19):
    holder_array = np.empty((0,41))
    for vset in range(len(allfinalarrays1)):
        tfinal_result = allfinalarrays1[vset]
        holder_array = np.concatenate((holder_array,tfinal_result[elec]))
    elec_wise_results1.append(holder_array)


elec_wise_results2=[]
for elec in range(19):
    holder_array = np.empty((0,41))
    for vset in range(len(allfinalarrays2)):
        tfinal_result = allfinalarrays2[vset]
        holder_array = np.concatenate((holder_array,tfinal_result[elec]))
    elec_wise_results2.append(holder_array)




# %% codecell

freq_bands = [(0,4),(4,8),(8,12),(12,16),(16,32),(32,42)]
log = True
elec_wise_fooof1=[]
for elec in range(19):
    holder_array=[]
    for j in range(elec_wise_results1[elec].shape[0]):
        ff = fooof.FOOOF(peak_width_limits=[
                         5, 20], background_mode='knee')
        try:
            ff.fit(frequencies,elec_wise_results1[elec][j] , freq_range=[0.1, endfoooffreq])
            if log:
                fooofdiff = ff.power_spectrum - ff._bg_fit
            else:
                fooofdiff = np.power(10,ff.power_spectrum) - np.power(10,ff._bg_fit)

            tempsum = np.zeros(len(freq_bands))
            for i in range(fooofdiff.shape[0]):
                freq = ff.freqs[i]
                for k in range(tempsum.shape[0]):
                    lowlimit = freq_bands[k][0]
                    highlimit = freq_bands[k][1]
                    if (freq > lowlimit)and(freq <= highlimit):
                        tempsum[k] = tempsum[k] + fooofdiff[i]
                        break

            holder_array.append(tempsum)
        except:
            a=1

    try:
        holder_array = np.stack(holder_array,axis=0)
    except:
        holder_array = np.empty((0,6))
    elec_wise_fooof1.append(holder_array)


elec_wise_fooof2=[]
for elec in range(19):
    holder_array=[]
    for j in range(elec_wise_results2[elec].shape[0]):
        ff = fooof.FOOOF(peak_width_limits=[
                         5, 20], background_mode='knee')
        try:
            ff.fit(frequencies,elec_wise_results2[elec][j] , freq_range=[0.1, endfoooffreq])

            if log:
                fooofdiff = ff.power_spectrum - ff._bg_fit
            else:
                fooofdiff = np.power(10,ff.power_spectrum) - np.power(10,ff._bg_fit)

            tempsum = np.zeros(len(freq_bands))
            for i in range(fooofdiff.shape[0]):
                freq = ff.freqs[i]
                for k in range(tempsum.shape[0]):
                    lowlimit = freq_bands[k][0]
                    highlimit = freq_bands[k][1]
                    if (freq > lowlimit)and(freq <= highlimit):
                        tempsum[k] = tempsum[k] + fooofdiff[i]
                        break

            holder_array.append(tempsum)
        except:
            a=1


    try:
        holder_array = np.stack(holder_array,axis=0)
    except:
        holder_array = np.empty((0,6))
    elec_wise_fooof2.append(holder_array)

# %% codecell
'''
elec_wise_fooof1=[]
for elec in range(19):
    holder_array=[]
    for j in range(elec_wise_results1[elec].shape[0]):
        ff = fooof.FOOOF(peak_width_limits=[
                         5, 20], background_mode='fixed')
        try:
            ff.fit(frequencies,elec_wise_results1[elec][j] , freq_range=[0.1, endfoooffreq])

            fooof_ori = np.power(10,ff._bg_fit)
            sub_f = fooof_ori[1:]
            fooof_ori = fooof_ori[:-1]
            sub = (fooof_ori-sub_f)/2.5

            holder_array.append(sub)
        except:
            a=1


    holder_array = np.stack(holder_array,axis=0)
    elec_wise_fooof1.append(holder_array)

elec_wise_fooof2=[]
for elec in range(19):
    holder_array=[]
    for j in range(elec_wise_results2[elec].shape[0]):
        ff = fooof.FOOOF(peak_width_limits=[
                         5, 20], background_mode='fixed')
        try:
            ff.fit(frequencies,elec_wise_results2[elec][j] , freq_range=[0.1, endfoooffreq])

            fooof_ori = np.power(10,ff._bg_fit)
            sub_f = fooof_ori[1:]
            fooof_ori = fooof_ori[:-1]
            sub = (fooof_ori-sub_f)/2.5

            holder_array.append(sub)
        except:
            a=1


    holder_array = np.stack(holder_array,axis=0)
    elec_wise_fooof2.append(holder_array)
'''
# %% codecell

meanarrays1 = []
for i in range(19):
    meanarrays1.append(np.mean(elec_wise_fooof1[i],axis=0))

meanarrays1 = np.stack(meanarrays1,axis=0)

meanarrays2 = []
for i in range(19):
    meanarrays2.append(np.mean(elec_wise_fooof2[i],axis=0))

meanarrays2 = np.stack(meanarrays2,axis=0)


# %% codecell
stdarrays1 = []
for i in range(19):
    stdarrays1.append(np.std(elec_wise_fooof1[i],axis=0))
stdarrays1 = np.stack(stdarrays1,axis=0)

stdarrays2 = []
for i in range(19):
    stdarrays2.append(np.std(elec_wise_fooof2[i],axis=0))
stdarrays2 = np.stack(stdarrays2,axis=0)

# %% codecell

from scipy.stats import ttest_ind_from_stats


# %% codecell
p_values =[]
for i in range(19):
    t,p=ttest_ind_from_stats(meanarrays1[i],stdarrays1[i],elec_wise_fooof1[i].shape[0],meanarrays2[i],stdarrays2[i],elec_wise_fooof2[i].shape[0],equal_var=False)
    p_values.append(p)
p_values=np.stack(p_values,axis=0)

# %% codecell
final_result1 = meanarrays1.T

final_result2 = meanarrays2.T


# %% codecel
store_name = 'final_res/'+sleep_stage+str(window_type)
with open(store_name+'fr1.data', 'wb') as filehandle:
    pickle.dump(final_result1, filehandle)
with open(store_name+'fr2.data', 'wb') as filehandle:
    pickle.dump(final_result2, filehandle)
with open(store_name+'pvalue.data', 'wb') as filehandle:
    pickle.dump(p_values, filehandle)
    # %% codecel

# %% codecel


# %% codecel

#sleep_stage='rem'
store_name = 'final_res/'+sleep_stage+str(window_type)

with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T


fdr_pvalues = []
for i in range(6):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)

p_values_bool = p_values<=0.05
fdr_pvalues_bool = fdr_pvalues<=0.05

fig = plt.figure(figsize =(10,10))
fig.suptitle("SLEEP STAGE : "+sleep_stage,fontsize=26)

count=1
maxall = max(final_result1.max(),final_result2.max())
minall = max(final_result1.min(),final_result2.min())



final_result = final_result1-final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(3,6,count)
    if i==2:
        ax.set_title('Good-bad windows',fontsize='xx-large')

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

final_result = final_result1
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(3,6,count)
    if i==2:
        ax.set_title('Good windows',fontsize='xx-large')

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                    extrapolate='skirt',
                    axes=ax,
                    vmax = maxall,
                    vmin =minall

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

final_result = final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(3,6,count)
    if i==2:
        ax.set_title('bad windows',fontsize='xx-large')

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = maxall,
                     vmin =minall

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

#plt.savefig(sleep_stage+'freq.pdf')
plt.show()



# %% codecel
'''
sleep_stage='rem'

store_name = 'final_res1/'+sleep_stage

with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T


fdr_pvalues = []
for i in range(15):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)
fdr_pvalues = fdr_pvalues
fdr_pvalues_bool = fdr_pvalues<=0.01


fig = plt.figure(figsize =(20,20))
count=1
fig.suptitle("Sleep stage: "+sleep_stage,fontsize=26)

final_result = np.zeros(19)
for i in range(15):
    ax = fig.add_subplot(4,4,count)
    ax.set_title('freq bin: '+str(i),fontsize='xx-large')


    ax,_=plot_topomap( final_result,
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
#cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)
#plt.savefig(sleep_stage+'.png')

plt.show()
'''
# %% codecel



# %% codecel


# %% codecel
window_type=2
fig = plt.figure(figsize =(15,10))
#fig.suptitle("All sleep stage ,HDR - LDR (good windows only) ",fontsize=26)

count=1

sleep_stage='awa'
caps_sleep_stage = 'AWA'
store_name = 'final_res-main/'+sleep_stage+str(window_type)

with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T


fdr_pvalues = []
for i in range(6):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)

p_values_bool = p_values<=0.05
fdr_pvalues_bool = fdr_pvalues<=0.05


maxall = max(final_result1.max(),final_result2.max())
minall = max(final_result1.min(),final_result2.min())



final_result = final_result1-final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(4,6,count)

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)


sleep_stage='s2'
caps_sleep_stage = 'S2'
store_name = 'final_res-main/'+sleep_stage+str(window_type)
with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T



fdr_pvalues = []
for i in range(6):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)

p_values_bool = p_values<=0.05
fdr_pvalues_bool = fdr_pvalues<=0.05


maxall = max(final_result1.max(),final_result2.max())
minall = max(final_result1.min(),final_result2.min())



final_result = final_result1-final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(4,6,count)

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

sleep_stage='sws'
caps_sleep_stage = 'SWS'
store_name = 'final_res-main/'+sleep_stage+str(window_type)
with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T



fdr_pvalues = []
for i in range(6):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)

p_values_bool = p_values<=0.05
fdr_pvalues_bool = fdr_pvalues<=0.05


maxall = max(final_result1.max(),final_result2.max())
minall = max(final_result1.min(),final_result2.min())



final_result = final_result1-final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(4,6,count)

    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

sleep_stage='rem'
caps_sleep_stage = 'REM'
store_name = 'final_res-main/'+sleep_stage+str(window_type)
with open(store_name+'fr1.data', 'rb') as filehandle:
    final_result1 = pickle.load(filehandle)
with open(store_name+'fr2.data', 'rb') as filehandle:
    final_result2 = pickle.load(filehandle)
with open(store_name+'pvalue.data', 'rb') as filehandle:
    p_values = pickle.load(filehandle)
p_values = p_values.T



fdr_pvalues = []
for i in range(6):
    #t= pvalues_correction(p_values[i],'fdr','indep')
    _,t = mne.stats.fdr_correction(p_values[i], alpha=0.05, method='indep')
    #_,t= fdrcorrection(p_values[i], alpha=0.05)
    fdr_pvalues.append(t)
fdr_pvalues = np.stack(fdr_pvalues,axis=0)

p_values_bool = p_values<=0.05
fdr_pvalues_bool = fdr_pvalues<=0.05


maxall = max(final_result1.max(),final_result2.max())
minall = max(final_result1.min(),final_result2.min())



final_result = final_result1-final_result2
for i in range(final_result.shape[0]):
    ax = fig.add_subplot(4,6,count)


    ax,_=plot_topomap( final_result[i],
                    SENSORS_POS,
                    show=False,
                    cmap="viridis",
                    names=CHANNEL_NAMES,
                    show_names=False,
                    contours=0,
                     extrapolate='skirt',
                     axes=ax,
                     vmax = final_result.max(),
                     vmin = final_result.min(),
                     mask = fdr_pvalues_bool[i],
                     mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='w',
                     linewidth=0, markersize=10)

                    )
    count=count+1
cb = fig.colorbar(ax, orientation="vertical",aspect=40,fraction=0.03)

#plt.savefig('main-fig.png',dpi=300)
plt.show()




# %% codecell


# %% codecell
