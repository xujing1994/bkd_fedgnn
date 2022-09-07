"""
    File to load dataset based on user control from main file
"""
import os
#os.chdir('../') # go to root folder of the pro
import sys
sys.path.append('/home/nfs/federated_learning_jx/federated_learning/GNN_common/data')

from TUs import TUsDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """    

    # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full', 'MUTAG', 'NCI1', 'MCF-7', \
            'MCF-7H', 'MOLT-4', 'MOLT-4H', 'NCI-H23', 'NCI-H23H', 'OVCAR-8', 'OVCAR-8H', \
            'COLLAB', 'deezer_ego_nets', 'github_stargazers', 'reddit_threads', 'twitch_egos', 'COLORS-3', 'TRIANGLES']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)
