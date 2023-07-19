#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:14:12 2022

@author: anagathil
"""
import torch

def comp(x,c,mode):
    if mode == 'symlog':
        y = torch.sign(x)*torch.log(1+c*torch.abs(x))
    elif mode == 'sqrt':
        y = torch.sign(x)*torch.pow(torch.abs(x),c)
    return y

def invcomp(y,c,mode):
    if mode == 'symlog':
        x = torch.sign(y)*(torch.exp(torch.abs(y))-1)/c
    elif mode == 'sqrt':
        x = torch.sign(y)*torch.pow(torch.abs(y),1/c);
    return x
    

def rescaleIhcograms(scaleType,scaleWeight,neurogram_pred,neurogram,normFactor):
    if scaleType == 'symlog' or scaleType == 'sqrt':
        neurogram_pred = invcomp(neurogram_pred*normFactor, scaleWeight, scaleType) 
        neurogram = invcomp(neurogram*normFactor, scaleWeight, scaleType)
    else:
        neurogram_pred = neurogram_pred*normFactor
        neurogram = neurogram*normFactor
    return neurogram_pred, neurogram

