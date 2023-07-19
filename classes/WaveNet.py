#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:29:23 2021

@author: anagathil
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNet(nn.Module):
    def __init__(self,nLayers,nStacks,nChannels,nResChannels,nSkipChannels,numOutputLayers):
        super(WaveNet, self).__init__()
        self.name = 'WaveNet'
        self.nChannels = nChannels
        self.nResChannels = nResChannels
        self.nSkipChannels = nSkipChannels
        self.nLayers = nLayers
        self.nStacks = nStacks
        self.kernelSize = 2
        self.nSkipOffset = 2**self.nLayers
        
        # intial conv layer
        self.conv0 = nn.Conv1d(in_channels=1,out_channels=self.nResChannels,kernel_size=1,stride=1,padding=0,dilation=1)  #edit      
        
        # stack of residual layers
        self.residualBlockList = nn.ModuleList()
        for iLayers in range(self.nLayers*self.nStacks):
            expval = iLayers % self.nLayers
            dilation = 2**expval
            self.residualBlockList.append(WaveNetUnit(self.nResChannels, self.nSkipChannels, dilation))
      
        # output layers
        self.numOutputLayers = numOutputLayers
        self.outputLayers = nn.ModuleList()
        self.outputActivations = nn.ModuleList()        
        for oLayers in range(self.numOutputLayers):
            self.outputActivations.append(nn.PReLU(num_parameters=self.nSkipChannels))
            
            if oLayers == self.numOutputLayers - 1:
                self.outputLayers.append(nn.Conv1d(in_channels=self.nSkipChannels, out_channels=self.nChannels, kernel_size=1))
            else:
                self.outputLayers.append(nn.Conv1d(in_channels=self.nSkipChannels, out_channels=self.nSkipChannels, kernel_size=1))
            
    def forward(self, input):
        input = torch.unsqueeze(input,dim = 0)
        input = torch.unsqueeze(input,dim = 0)
        x = self.conv0(input)
        
        self.nSkipOffset = (2**self.nLayers)*self.nStacks
        for iLayers in range(self.nLayers*self.nStacks):
            self.nSkipOffset = self.nSkipOffset - 2**(iLayers % self.nLayers)
            x, x_skip = self.residualBlockList[iLayers](x,self.nSkipOffset)
            if iLayers == 0:
                skipSum = x_skip
            else:
                skipSum = skipSum + x_skip
        
        out = skipSum
        for oLayers in range(self.numOutputLayers):
            out = self.outputActivations[oLayers](out)
            out = self.outputLayers[oLayers](out)
        
        return out
        
class WaveNetUnit(nn.Module):
    def __init__(self, nResChannels, nSkipChannels, filter_dilation):
        super(WaveNetUnit, self).__init__()
        self.conv_filter = nn.Conv1d(in_channels=nResChannels,out_channels=nResChannels,kernel_size=2,stride=1,padding=0,dilation=filter_dilation)    
        self.conv_gate = nn.Conv1d(in_channels=nResChannels,out_channels=nResChannels,kernel_size=2,stride=1,padding=0,dilation=filter_dilation)
        
        self.conv1x1_res = nn.Conv1d(in_channels=nResChannels, out_channels=nResChannels, kernel_size=1)
        self.conv1x1_skip = nn.Conv1d(in_channels=nResChannels, out_channels=nSkipChannels, kernel_size=1)
        
        self.filter_tanH = nn.Tanh()
        # self.filter_tanH = utils_neur.slclass() #'NEW': 11.05.2022
        self.gate_sigmoid = nn.Sigmoid()
        
    def forward(self, x, skipOffset):
        # print(x.shape)
        
        x_filter = self.filter_tanH(self.conv_filter(x))
        x_gate = self.gate_sigmoid(self.conv_gate(x))

        # print(x_filter.shape)
        # print(x_gate.shape)

        x_res = self.conv1x1_res(x_filter*x_gate)
        x_skip = self.conv1x1_skip(x_filter*x_gate)
        x_skip = x_skip[:,:,skipOffset-1:]
        # print(x_skip.shape)

        # print(x_res.shape)
        # print(x_skip.shape)
        
        # x_res = self.BatchNormRes(x_res) #new
        x_res = x_res + x[:,:,-x_res.shape[2]:]
       
        return x_res, x_skip

if __name__ == "__main__":

     input1 = torch.randn((1,1,16000))
     # input1 = torch.unsqueeze(input1,1)
     print(input1.shape)
     
     net = WaveNet(nLayers= 8,nStacks=4, nChannels = 80,nResChannels = 80,nSkipChannels = 80,numOutputLayers=2)
     y = net(input1)
     print(y.shape)
     
     pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
     print('Number of parameters: ' + str(pytorch_total_params))

