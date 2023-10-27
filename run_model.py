#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:23:39 2020

@author: anagathil
"""
import time
import torch
import numpy as np
from classes import WaveNet
from utils import utils
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from scipy.io import savemat 
import yaml
import os
import librosa
from argparse import ArgumentParser

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, default='./audio/61-70968-0000.flac', help='File to be processed.')
    parser.add_argument("--spl", type=float, default=60, help='Sound pressure level, at which signal is processed.')
    parser.add_argument("--totdur", type=float, default=4, help='Segment duration.')
    parser.add_argument("--segdur", type=float, default=1, help='Segment duration.')
    parser.add_argument("--model", type=str, default="./model/musan31rfa3-1fullSet_20231014-145738.pt", help='Path to model parameters (.pt file)')
    parser.add_argument("--config", type=str, default="./config/config31rfa3-1fullSet.yaml", help='Path to config file (.yaml file)')
    parser.add_argument("--proc", type=str, choices=["cpu","cuda"], default="cpu", help='Choose cpu or gpu processing.')
    parser.add_argument("--matlabComp", type=int, choices=[0,1], default=0, help='Enable/disable comparison with original MATLAB/C auditory model code.')
    args = parser.parse_args()

    # load configuration file  
    with open(args.config,'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)   

    # constants
    sigMax = torch.tensor(55)
    ihcogramMax = torch.tensor(1.33)
    ihcogramMax = utils.comp(ihcogramMax, conf['scaleWeight'], conf['scaleType'])
    fs = 16000
    
    # number of samples to be skipped due to WaveNet processing    
    skipLength = (2**conf['nLayers'])*conf['nStacks']               

    # select processing device (either cpu or cuda)
    device = torch.device(args.proc)
    print('running on: ', device)
    
    ## initialize WaveNet and load model paramaters
    NET = WaveNet.WaveNet(conf['nLayers'],
                          conf['nStacks'],
                          conf['nChannels'],
                          conf['nResChannels'],
                          conf['nSkipChannels'],
                          conf['numOutputLayers'])
    NET.to(device)
    NET.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
    
    # compute number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in NET.parameters() if p.requires_grad)
    print('Number of parameters: ' + str(pytorch_total_params))
                       
    # read and normalize signal
    signal, _ = librosa.load(args.file, sr=fs, duration=args.totdur)    
    signal = signal/np.std(signal)*20e-6*10**(args.spl/20)
    signal = signal/sigMax    
    signal = signal.to(device)

    # segmentation
    frame_shift = int(args.segdur*fs)
    frame_len = frame_shift + skipLength
    sigLen = signal.shape[0]
    
    ihcogram_pred = np.zeros((80,sigLen-skipLength+1))
    
    start_idx = 0
    end_idx = 0
    frame_idx = 0
        
    print("Processing Started...")
     
    start_time = time.time()
    
    # loop over frames
    while end_idx < sigLen:
        start_idx = frame_idx*frame_shift
        end_idx = np.min([sigLen,frame_idx*frame_shift + frame_len])

        if sigLen - end_idx < skipLength:
            end_idx = sigLen
        signal_seg = signal[start_idx:end_idx]
        signal_seg = signal_seg.to(device)
                     
        # process signal segment and scale back to original range
        ihcogram_pred_seg = NET(signal_seg)
        ihcogram_pred_seg = ihcogram_pred_seg*ihcogramMax    
        ihcogram_pred_seg = utils.invcomp(ihcogram_pred_seg, conf['scaleWeight'], conf['scaleType'])

        # write IHCogram segment into full IHCogram array
        ihcogram_pred_seg = ihcogram_pred_seg.cpu().detach().numpy()
        ihcogram_pred[:,start_idx:end_idx-skipLength+1] = ihcogram_pred_seg 

        frame_idx += 1
    
    end_time = time.time()
    
    elapsed = end_time-start_time
    print('Processing time of WaveNet model: ' + str(elapsed) + ' s')
    
    # save IHCogram for MATLAB comparison
    if args.matlabComp == 1:
        savemat('tmp.mat',{'ihcogram': ihcogram_pred})
    
    if 1==0:

        climlow = -0.0025
        climhigh = 0.06
                    
        plt.rcParams.update({'font.family':'serif'})
        plt.rcParams.update({'font.size':22})
        # plt.rcParams['axes.axisbelow'] = True

        fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(8,7))
        fig.tight_layout()        
        axes[0].imshow(ihcogram_pred,cmap='gray_r', aspect='auto', extent=[0,max_t,ihcogram_in.shape[0],1], vmin=climlow,vmax=climhigh)
        axes[0].invert_yaxis()
        axes[0].set_xticklabels('')
        axes[0].set_yticks(np.arange(0,81,20))
        axes[0].set_ylabel('CF index')
        axes[0].set_title('Approximation',fontsize=20)
        fig.colorbar(orig,ax=axes[:],location='right',aspect=50,pad=0.02)

                
    

