
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
   ####################              DATASET              ####################
   ###########################################################################

# utils
def read_in_csv(data_fname):
    # read in data such that each row is a list of dicts
    with open(data_fname, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        d = [] # list of dicts (one per training item)
        for line in reader:
            d.append(dict(zip(headers, [line[k] for k in headers])))
        return d
    
def to_np(t):
    return t.data.numpy()

class Dataset(object):
    def __init__(self, data_fname="data.csv"):
        self.train_data = self.format_dataset(read_in_csv(data_fname))
        self.train_size = len(self.train_data["names"])
        self.test_data = self.train_data
        self.test_size = len(self.test_data["names"])
        
    def format_dataset(self, d_in):
        d = {
                "inputs": {
                     # list of summand, where each summand is a num_digits one-hot 
                     # tensor
                    "in1": [], 
                     # list of relations as num_digits one-hot tensor
                    "in2": []
                }, 
                # list of targets, where each target is a two num_digits tensor
                "targets": [], 
                # list of name strings, e.g. "one"
                "names": [],
                "in1_names": [],
                "in2_names": []
            } 
        digit_names = sorted(list(set([training_item["in1"]
                                      for training_item  in d_in])))

        # Similar to digit names, except also has ten and over
        out_names = sorted(list(set([training_item["out2"]
                                    for training_item in d_in])))

        self.num_digits = len(digit_names)

        self.num_outs = len(out_names)

        # assign inds to names and create lookups in both directions for easy 
        # reference later
        self.digit_names_to_inds = dict(zip(digit_names, 
                                      list(range(self.num_digits))))

        self.digit_inds_to_names = dict(zip(list(range(self.num_digits)),
                                      digit_names))

        self.out_names_to_inds = dict(zip(out_names,
                                     list(range(self.num_outs))))

        self.out_inds_to_names = dict(zip(list(range(self.num_outs)),
                                      out_names))

        # package
        for training_item in d_in:
            in1_one_hot = self.digit_name_to_one_hot(training_item["in1"])
            in2_one_hot = self.digit_name_to_one_hot(training_item["in2"])
            d["inputs"]["in1"].append(in1_one_hot)
            d["inputs"]["in2"].append(in2_one_hot)
            out_one_hot = torch.cat((
                    self.out_name_to_one_hot(training_item["out1"]),
                    self.out_name_to_one_hot(training_item["out2"])), 0)
            d["targets"].append(out_one_hot)
            # for visualization GUI only
            d["names"].append("{} {}".format(training_item["in1"], 
                                             training_item["in2"]))
            d["in1_names"].append(training_item["in1"])
            d["in2_names"].append(training_item["in2"])
        return d

    def digit_name_to_one_hot(self, digit_name):
        one_hot = torch.zeros(self.num_digits)
        one_hot[self.digit_names_to_inds[digit_name]] = 1.
        return one_hot

    def one_hot_to_digit_name(self, one_hot):
        ind = int(to_np(torch.nonzero(one_hot)))
        return self.digit_inds_to_names[ind]
   
    def out_name_to_one_hot(self, out_name):
        one_hot = torch.zeros(self.num_outs)
        one_hot[self.out_names_to_inds[out_name]] = 1.
        return one_hot

    def one_hot_to_out_name(self, one_hot):
        ind = int(to_np(torch.nonzero(one_hot)))
        return self.out_inds_to_names[ind]
   
    ##############################
    # UNUSED AND CURRENTLY WRONG #
    ##############################
    def print_batch(self, batch_inputs, batch_targets, predictions=None, 
                    targets_as_names=False):
        for ix, batch_item in enumerate(batch_inputs["item"]): TODO
        for batch_item_ind in range(len(batch_inputs["item"])):
            print("\n")
            print("Input: {}, {}".format(
                    self.one_hot_to_name(
                        batch_inputs["item"][batch_item_ind], "item"),
                    self.one_hot_to_name(
                        batch_inputs["relation"][batch_item_ind], "relation")))
            if targets_as_names:
                print(batch_targets[batch_item_ind])
                print("Target: {}".format(self.attribute_pattern_to_names(
                                              batch_targets[batch_item_ind])))
            else:
                print("Target: {}".format(to_np(batch_targets[batch_item_ind])))
            if not predictions is None:
                print("Model output: {}".format(
                    to_np(predictions[batch_item_ind, :])))
    
    def prepare_batch(self, batch, batch_size):
        # stack batch items to create batch_size x num_items, 
        # batch_num_relations, and batch_size x num_attributes tensors
        inputs = {"in1": torch.stack(batch["inputs"]["in1"], 0),
                  "in2": torch.stack(batch["inputs"]["in2"], 0)}
        targets = torch.stack(batch["targets"], 0)
        return (inputs, targets)
            
    def get_per_epoch_batches(self, batch_size):
        assert self.train_size % batch_size == 0
        train_item_inds = list(range(self.train_size))
        random.shuffle(train_item_inds)
        batches = []
        
        for batch_start in np.arange(0, self.train_size, batch_size):
            batch_inds = train_item_inds[batch_start : batch_start + batch_size]
            # pull selected inds from train set
            batch = {"inputs": 
                        {"in1": [self.train_data["inputs"]["in1"][i]
                                  for i in batch_inds], 
                         "in2": [self.train_data["inputs"]["in2"][i]
                                  for i in batch_inds]
                        },
                     "targets": [self.train_data["targets"][i]
                                  for i in batch_inds]
                    }
            # reformat to be tensors
            batches.append(self.prepare_batch(batch, batch_size)) 
        return batches
    
