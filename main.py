
   ###########################################################################
   ####################               SETUP               ####################
   ###########################################################################
    
# klh 1/2019
import sys
print(sys.version) # ensure that you're using python3
import torch
print("PyTorch version = {}".format(torch.__version__)) # ensure you're using 1.0.0
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

from dataset import *
from model import *
from trainer import *


def main():    
    # PARAMETERS 
    # training params
    num_training_epochs = 3000 # 3000
    stopping_criterion = 0.00 # 0.0
    train_batch_size = 1 # 1
    
    # optimization params
    learning_rate = 0.1 # 0.01
    
    # model params
    hidden_nonlinearity = "sigmoid" # "sigmoid"
    
    # initialization params
    weight_initialization_range = [-0.1, 0.1] # [-0.1, 0.1]
    output_layer_bias_val = -2. # -2.
    freeze_output_layer_biases = True # True
    
    # create dataset
    data_fname = "dir_data/data_minus_2.csv" 
    dataset = Dataset(data_fname=data_fname)
    
    # create model
    # some model and train params are locked to dataset params
    in1_size = len(dataset.digit_names_to_inds)
    in2_size = len(dataset.digit_names_to_inds)
    hidden_size = 30 # 15
    output_size = len(dataset.out_names_to_inds)
    
    model = Model(in1_size=in1_size,
                  in2_size=in2_size,
                  hidden_size=hidden_size, 
                  output_size=output_size, 
                  hidden_nonlinearity=hidden_nonlinearity,
                  wrange=weight_initialization_range, 
                  output_layer_bias_val=output_layer_bias_val,
                  freeze_output_layer_biases=freeze_output_layer_biases)


    # set up result dirs
    make_dir("logdirs")
    save_dir = get_logdir()
    print('Results will be saved in: {}'.format(save_dir))
    
    # create trainer
    trainer = Trainer(dataset, model, 
                      name="{}".format(0),
                      train_batch_size=train_batch_size,
                      num_training_epochs=num_training_epochs,
                      stopping_criterion=stopping_criterion,
                      learning_rate=learning_rate, 
                      save_dir=save_dir,
                      test_freq=10,
                      print_freq=100,
                      show_plot=True)
    trainer.train()
    
main()
