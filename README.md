# More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks

This repository is the official PyTorch implementation of the experiments in the following paper: 

Jing Xu*, Rui Wang, Stefanos Koffas, Kaitai Liang and Stjepan Picek. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks . ACSAC 2022. 

[arXiv](https://arxiv.org/abs/2202.03195)

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
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.9.0+cpu and 1.9.0+cu111 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```
## Test run
Test distributed backdoor attack
```
python dis_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 2 --filename ./Results/DBA
```
Test centralized backdoor attack
```
python cen_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/config/TUS/TUs_graph_classification_GCN_MUTAG_100k.json --num_workers 5 --num_mali 2 --filename ./Results/CBA
```


