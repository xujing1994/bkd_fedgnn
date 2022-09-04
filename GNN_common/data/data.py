"""
    File to load dataset based on user control from main file
"""
import os
#os.chdir('../') # go to root folder of the pro
import sys
sys.path.append('/home/nfs/federated_learning_jx/federated_learning/GNN_common/data')

from superpixels import SuperPixDataset
from molecules import MoleculeDataset
from TUs import TUsDataset
from SBMs import SBMsDataset
from TSP import TSPDataset
from COLLAB import COLLABDataset
from CSL import CSLDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full', 'MUTAG', 'NCI1', 'MCF-7', 'MCF-7H', 'MOLT-4', 'MOLT-4H', 'NCI-H23', 'NCI-H23H', 'OVCAR-8', 'OVCAR-8H', 'COLLAB', 'deezer_ego_nets', 'github_stargazers', 'reddit_threads', 'twitch_egos', 'COLORS-3', 'TRIANGLES']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)
    
    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    # handling for the CSL (Circular Skip Links) Dataset
    if DATASET_NAME == 'CSL': 
        return CSLDataset(DATASET_NAME)
    
