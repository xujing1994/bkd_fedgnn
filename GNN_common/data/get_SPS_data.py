import numpy as np
import torch
import pickle
import time
import os
#matplotlib inline
import matplotlib.pyplot as plt

import pickle

#load_ext autoreload
#autoreload 2

from superpixels import SuperPixDatasetDGL 

from data import LoadData
from torch.utils.data import DataLoader
from superpixels import SuperPixDataset
from superpixels import DGLFormDataset

import argparse

if __name__ ==  '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help="Please give a config.json file with training/model/data/param details")
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    dataset = SuperPixDatasetDGL(DATASET_NAME) 

    print('Time (sec):',time.time() - start) # 356s=6min

    start = time.time()

    with open('/tudelft.net/staff-umbrella/GS/Graph_Neural_Networks/federated_learning_jx/data/superpixels/%s_partition.pkl'%args.dataset_name,'wb') as f:
            pickle.dump([dataset.partition,dataset.test],f)
            
    print('Time (sec):',time.time() - start) # 38s

    
