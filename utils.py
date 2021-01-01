import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pyrubberband as pyrb

from tqdm import tqdm 
from glob import glob
from chords import Chords
from nnAudio import Spectrogram
from torchaudio.transforms import ComplexNorm
from torch.nn import ConstantPad1d
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit


def data_pre(augm=False,batch_size=32):
    print('data augmentation =',augm)
    
    file = './tensor_dict.pt'
    hift_factors = [0]
    
    if augm: 
        file = './tensor_dict_augm.pt'
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    
    if not os.path.isfile(file):
        print('data need process')
        sr = 44100
        window_size = 3
        shift_size = 0.5
        names = [file.split('/')[1][0:3] for file in sorted(glob('audio/*'))]

        g = 0
        Chord_class = Chords()
        xs = torch.ShortTensor([])
        ys = torch.ShortTensor([])
        group = np.array([])
        
        with open('dic.json') as json_file:
            dic = json.load(json_file)
        for name in tqdm(names[:5]):
            waveform_, sample_rate = torchaudio.load('audio/'+name+'.mp3')
            chord_info = Chord_class.get_converted_chord_voca('Labels/'+name+'.txt') #only maj min /7
            #pading zero in sizr window_size/2 on start & end 
            pad_len = (window_size/2)*sr
            pad = ConstantPad1d(int(pad_len),0)
            waveform = pad(waveform_[0])
            
            for shift_factor in shift_factors:
                
                if augm:
                    waveform = torch.ShortTensor(pyrb.pitch_shift(waveform, sr, shift_factor))
                

                #make a sliding window 
                x = waveform.unfold(dimension = 0,
                                         size = int(window_size*sr),
                                         step = int(shift_size*sr)).unsqueeze(1)
                
                #get labels
                chords = []
                i = 0
                if chord_info['start'][0] != 0 : i = chord_info['start'][0]
                for c in range(len(chord_info)):  
                    while chord_info['start'][c] <= i < chord_info['end'][c]:
                        #print(start,end,i,typs)
                        chord = chord_info['chord_id'][c]
                        if chord != 169 and chord != 168:
                            chord += shift_factor * 14
                            chord = chord % 168
                        chords.append(dic[str(chord)])
                        i += shift_size
                y = torch.ShortTensor(chords)

                #make label and musiz len is same
                if x.shape[0] > y.shape[0]:
                    x = x[:y.shape[0]]
                if y.shape[0] > x.shape[0]:
                    y = y[:x.shape[0]]

                #make a group for GroupShuffleSplit
                group = np.append(group,np.repeat(g,y.shape[0]))
                g += 1

                #extand 
                xs = torch.cat((xs,x))
                ys = torch.cat((ys,y))   

        #save
        dic = {'xs':xs,'ys':ys,'group':group}
        if augm:
            torch.save(dic, 'tensor_dict_augm.pt')
        else :
            torch.save(dic, 'tensor_dict.pt')
    
    print('data do not need process')
    dic = torch.load('tensor_dict.pt')
    xs,ys,group = [dic[c] for c in dic]
    gss = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)
    for train_index, test_index in gss.split(xs, ys, group):
        X_train, X_test = xs[train_index], xs[test_index]
        y_train, y_test = ys[train_index], ys[test_index]

    dataloader_X_train = DataLoader(X_train,batch_size=32,shuffle=False, num_workers=0,drop_last=True)
    dataloader_y_train= DataLoader(y_train,batch_size=32,shuffle=False, num_workers=0,drop_last=True)

    dataloader_X_test = DataLoader(X_test,batch_size=32,shuffle=False, num_workers=0,drop_last=True)
    dataloader_y_test= DataLoader(y_test,batch_size=32,shuffle=False, num_workers=0,drop_last=True)

    return dataloader_X_train,dataloader_y_train,dataloader_X_test,dataloader_y_test

def test_cqt_2010(x):
    # Log sweep case
    fs = 44100
    #x = x.astype(dtype=np.float32)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Magnitude
    stft = Spectrogram.CQT2010(sr=fs,verbose=False).to(device)
    X = stft(torch.tensor(x,device=device))  
    return X

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.complex_norm = ComplexNorm(power=2.) #Compute the norm of complex tensor input. Power of the norm. (Default: to 1.0)
        
        cnn = nn.Sequential()
        cnn.add_module('conv{0}',   nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{0}',   nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{0}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{0}',nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{0}',   nn.Dropout(p=0.1))
                       
        cnn.add_module('conv{1}',   nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))
        cnn.add_module('norm{1}',   nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{1}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{1}',nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{1}',   nn.Dropout(p=0.1))
     
        cnn.add_module('conv{2}',   nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]))        
        cnn.add_module('norm{2}',   nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        cnn.add_module('relu{2}',   nn.ELU(alpha=1.0))
        cnn.add_module('pooling{2}',nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False))
        cnn.add_module('drop{2}',   nn.Dropout(p=0.1))
        self.cnn=cnn
        
        self.LSTM        = nn.LSTM(input_size = 4,hidden_size = 64,num_layers=2) #input_size change buy windows size
        self.Dropout     = nn.Dropout(p=0.1)
        self.BatchNorm1d = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.Linear      = nn.Linear(in_features=64 , out_features=10, bias=True)
        self.Linear2     = nn.Linear(in_features=640 , out_features=128, bias=True)
        self.Linear3     = nn.Linear(in_features=128 , out_features=50, bias=True)  #category number 
    
    def forward(self, x):                      #(batch,1,seq)
        x   = test_cqt_2010(x)[:,:,:,0].unsqueeze(1)    #(batch,featrue,seq,phase_real/phase_imag)
        x   = self.cnn(x)                      #(batch,chanel,featrue,seq)
        x   = x.flatten(start_dim=1,end_dim=2) #(batch,chanel*featrue,seq)
        x   = x.transpose(0,1)                 #(seq,batch,chanel*featrue)
        x,_ = self.LSTM(x)                     #(seq,batch,64)
        x   = self.Dropout(x)                  #(seq,batch,64)
        x   = x.transpose(0,1)                 #(batch,seq,64)
        x   = x.transpose(1,2)                 #(batch,64,seq)
        x   = self.BatchNorm1d(x)              #(batch,64,seq)
        x   = x.transpose(1,2)                 #(batch,seq,64)  
        x   = self.Linear(x)                   #(batch,seq,10)
        x   = x.flatten(start_dim=1)           #(batch,seq*10)
        x   = self.Linear2(x)                  #(batch,128)
        x   = self.Linear3(x)                  #(batch,10)
        return F.softmax(x)                    #(batch,10)
    
def adjusting_learning_rate(optimizer, factor=.5, min_lr=0.00001):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * factor, min_lr)
        param_group['lr'] = new_lr
        print('adjusting learning rate from %.6f to %.6f' % (old_lr, new_lr))
    