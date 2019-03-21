
   ###########################################################################
   ####################               SETUP               ####################
   ###########################################################################
    
# klh 1/2019
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import random
import matplotlib.pyplot as plt
# %matplotlib notebook
from scipy import spatial
import os
import pickle
import copy
import csv


   ###########################################################################
   ####################               MODEL               ####################
   ###########################################################################

class Model(nn.Module):
    def __init__(self,
                 in1_size, 
                 in2_size, 
                 hidden_size, 
                 output_size, 
                 hidden_nonlinearity="sigmoid",
                 final_nonlinearity="sigmoid",
                 wrange=[-0.9, 0.9],
                 output_layer_bias_val=-2.,
                 freeze_output_layer_biases=True):
        super(Model, self).__init__()
        # specify network layers
        self.linear_layers = nn.ModuleList(
                                [nn.Linear(in1_size + in2_size,
                                           hidden_size),
                                 nn.Linear(
                                     hidden_size,
                                     output_size)])
        self.nonlinearities = [self.select_nonlinearity(
                                   hidden_nonlinearity), 
                               self.select_nonlinearity(
                                   final_nonlinearity)] 
        self.layer_names = ["hidden_layer",
                            "output_layer"]

        self.trainable_layer_names = ["hidden_layer",
                                      "output_layer"]

    
        # initialize weights
        for layer, layer_name in zip(self.linear_layers, self.layer_names):
            self.init_weights(layer, layer_name, wrange, 
                              output_layer_bias_val, 
                              freeze_output_layer_biases=
                                  freeze_output_layer_biases)
            
    def select_nonlinearity(self, nonlinearity):
        if nonlinearity == "sigmoid":
            return nn.Sigmoid()
        elif nonlinearity == "tanh":
            return nn.Tanh()
        elif nonlinearity == "relu":
            return nn.ReLU()
        elif nonlinearity == "none":
            return lambda x: x
        
    def init_weights(self, layer, layer_name, wrange, output_layer_bias_val, 
                     freeze_output_layer_biases=True):
        layer.weight.data.uniform_(wrange[0], wrange[1]) # inplace
        if layer_name == "output_layer":
            if not output_layer_bias_val is None:
                layer.bias.data = (output_layer_bias_val *
                                   torch.ones(layer.bias.data.shape))
            else:
                layer.bias.data.uniform_(wrange[0], wrange[1])
            if freeze_output_layer_biases:
                layer.bias.requires_grad = False
        else:
            layer.bias.data.uniform_(wrange[0], wrange[1])
            
    def print_model_layers(self):
        print("\nModel linear layers:")
        print(self.linear_layers)
        print("\nModel nonlinearities:")
        print(self.nonlinearities)
        print("")

    def record_gnet(self, grad):
        self.gnet.append(grad)
        
    def record_gact(self, grad):
        self.gact.append(grad)

    def forward(self, inp, record_data=True):
        if record_data:
            # for visualization purposes; not necessary to train the model 
            self.layer_inputs = []
            self.layer_outputs = []
            self.layer_activations = []
            # these are recorded in backward order (the order of the gradient 
            # computations)
            self.gnet, self.gact = [], [] 
        
        for i, (layer, nonlinearity) in enumerate(zip(self.linear_layers, 
                                                      self.nonlinearities)):
            
            if i == 0:
                # input to first layer is just the item
                out = torch.cat((inp["in1"], inp["in2"]), 1)
            # visualization purposes only
            if record_data: 
                self.layer_inputs.append(out)

            out = layer(out)
            
            if record_data:
                self.layer_outputs.append(out)
                out.register_hook(self.record_gnet) # dE/dnet
                
            # apply nonlinearity
            out = nonlinearity(out)
            
            if record_data:
                self.layer_activations.append(out)
                out.register_hook(self.record_gact) # dE/da
        return out           
