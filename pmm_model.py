#
# This is the model for the following addition algorithm:
# Try adding In1, In2, resul is Add1
# Represent Add1 as Add1Rep, subtract In1 from Add1Rep, result is Sub1
# Subtract Sub1 from In2, result is sub2
# represent output from Add1, Sub2
#



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

ADD_MODULE_FILEPATH = "logdirs/addition_module_2.pkl"
SUB_MODULE_FILEPATH = "logdirs/subtraction_module_2.pkl"


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
                                     in1_size),
                                 nn.Linear(
                                     in1_size,
                                     in1_size),
                                 nn.Linear(
                                     in1_size + in1_size,
                                     hidden_size),
                                 nn.Linear(
                                     hidden_size,
                                     in1_size),
                                 nn.Linear(
                                     in2_size + in1_size,
                                     hidden_size),
                                 nn.Linear(
                                     hidden_size,
                                     in1_size),
                                 nn.Linear(
                                     in1_size + in1_size,
                                     output_size)])
        self.nonlinearities = [self.select_nonlinearity(
                                   hidden_nonlinearity)] * 8
        self.layer_names = ["a1h",
                            "a1o",
                            "a1r",
                            "s1h",
                            "s1o",
                            "s2h",
                            "s2o",
                            "out"]
        # Important: trainable layer names must be in order!!
        # e.g. ["a1r", "out"] \neq ["out", "a1r"]
        self.trainable_layer_names = ["a1r", "out"]
    
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
        with open(ADD_MODULE_FILEPATH, 'rb') as pkl:
            add_module = pickle.load(pkl)
        with open(SUB_MODULE_FILEPATH, 'rb') as pkl:
            sub_module = pickle.load(pkl)
        ahw = torch.from_numpy(add_module["hidden_layer"]["weights"])
        ahb = torch.from_numpy(add_module["hidden_layer"]["biases"])
        aow = torch.from_numpy(add_module["output_layer"]["weights"])
        aob = torch.from_numpy(add_module["output_layer"]["biases"])
        shw = torch.from_numpy(sub_module["hidden_layer"]["weights"])
        shb = torch.from_numpy(sub_module["hidden_layer"]["biases"])
        sow = torch.from_numpy(sub_module["output_layer"]["weights"])
        sob = torch.from_numpy(sub_module["output_layer"]["biases"])
        if layer_name == "a1h":
            layer.weight.data = ahw
            layer.bias.data = ahb
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        elif layer_name == "a1o":
            layer.weight.data = aow
            layer.bias.data = aob
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        elif layer_name in ["s1h", "s2h"]:
            layer.weight.data = shw
            layer.bias.data = shb
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        elif layer_name in ["s1o", "s2o"]:
            layer.weight.data = sow
            layer.bias.data = sob
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
        else:
            assert layer_name in self.trainable_layer_names
            layer.weight.data.uniform_(wrange[0], wrange[1]) # inplace
            if layer_name == "out":
                if not output_layer_bias_val is None:
                    layer.bias.data = (output_layer_bias_val *
                                       torch.ones(layer.bias.data.shape))
                else:
                    layer.bias.data.uniform_(wrange[0], wrange[1])
                if freeze_output_layer_biases:
                    layer.bias.requires_grad = False
            elif layer_name == "a1r":
                layer.bias.data = (output_layer_bias_val *
                                   torch.ones(layer.bias.data.shape))
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
        a1o = None
        in1 = inp["in1"]
        in2 = inp["in2"]
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
                out = torch.cat((in1, in2), 1)
            elif i == 3:
                # input to s1h is a1r concat in1
                out = torch.cat((out, in1), 1)
            elif i == 5:
                # Input to s2h is in2 concat s1o
                out = torch.cat((in2, out), 1)
            elif i == 7:
                # Input to out is s20 concat a1o
                out = torch.cat((out, a1o), 1)

            # visualization purposes only
            if record_data: 
                self.layer_inputs.append(out)

            out = layer(out)
            
            if record_data and i not in [0, 1, 3, 4, 5, 6]:
                self.layer_outputs.append(out)
                out.register_hook(self.record_gnet) # dE/dnet
                
            # apply nonlinearity
            out = nonlinearity(out)
            
            if record_data and i not in [0, 1, 3, 4, 5, 6]:
                self.layer_activations.append(out)
                out.register_hook(self.record_gact) # dE/da

            if i == 1:
                a1o = copy.deepcopy(out)
        return out           
