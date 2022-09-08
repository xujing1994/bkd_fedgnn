# More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks

This repository is the official PyTorch implementation of the experiments in the following paper: 

Jing Xu*, Rui Wang, Stefanos Koffas, Kaitai Liang and Stjepan Picek. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks . ACSAC 2022. [arXiv](https://arxiv.org/abs/2202.03195)

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).
```
@article{xu2022more,
  title={More is better (mostly): On the backdoor attacks in federated graph neural networks},
  author={Xu, Jing and Wang, Rui and Liang, Kaitai and Picek, Stjepan},
  journal={arXiv preprint arXiv:2202.03195},
  year={2022}
}
```

## Installation
Install PyTorch following the instuctions on the [PyTorch](https://pytorch.org/). The code has been tested over PyTorch 1.9.0+cpu and 1.9.0+cu111 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```
## Dataset
The dataset can be specified by setting '--dataset' with dataset name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``. The dataset name can be those on [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).

## Test run
1. Train a clean Federated GNN model
```
python clean_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 0 --filename ./Results/Clean
```
2. Test distributed backdoor attack in Federated GNNs
```
python dis_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 2 --filename ./Results/DBA
```
3. Test centralized backdoor attack in Federated GNNs
```
python cen_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/config/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 2 --filename ./Results/CBA
```
4. Test defense against backdoor attack in Federated GNNs

Here two defenses can be tested against the backdoor attack in Federated GNNs, by setting value of '--defense' to be 'foolsgold' or 'flame'.

Examples:
```
python dis_bkd_fedgnn.py --defense foolsgold --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 2
```
> Note: The experimental results won't be saved without value for `--filename`

> Note: In order to make sure trigger pattern in CBA is the union set of local trigger patterns in DBA, DBA should be implemented before the CBA. The reason can be found in the last paragraph of section 4.1 in the paper.

## Detailed Usage
### Configuration file
All arguments in the parer are able to set default values in the configuration file in ```./Common/Utils/options.py```

### Some important arguments
> Dataset name: `--dataset` (default='NCI1', help='name of dataset')
> Dataset path: `--datadir` (default=`./Data`, help=`path to save the downloaded dataset`)  
> Number of clients: `--num_workers` ()  

