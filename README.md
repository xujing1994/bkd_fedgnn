# More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks

This repo contais the source code used in our paper. It consists of 3 main
python scripts used for different tasks. In particular, `clean_fedgnn.py`
trains federated GNNs without any backdoors, `dis_bkd_fedgnn.py` trains
backdoored federated GNNs under the distributed backdoor attack (DBA) setup,
and `cen_bkd_fedgnn.py` trains backdoored federated GNNs under the centralized
backdoor atttack (CBA) setup.

## Installation
Install PyTorch following the instuctions on the [PyTorch](https://pytorch.org/). The code has been tested over `PyTorch 1.9.0+cpu` and `1.9.0+cu111` versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```
## Dataset
The dataset can be specified by setting `--dataset` with dataset name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``. The dataset name can be those on [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).

## Test run
1. Train a clean Federated GNN model
```
python clean_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 0 --filename ./Results/Clean
```
2. Test distributed backdoor attack in Federated GNNs
```
python dis_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2 --filename ./Results/DBA
```
3. Test centralized backdoor attack in Federated GNNs
```
python cen_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/config/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2 --filename ./Results/CBA
```
4. Test defense against backdoor attack in Federated GNNs

Here two defenses can be tested against the backdoor attack in Federated GNNs, by setting value of '--defense' to be 'foolsgold' or 'flame'.

Examples:
```
python dis_bkd_fedgnn.py --defense foolsgold --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2
```
> Note: The experimental results won't be saved without value for `--filename`.
> Note: In order to make sure trigger pattern in CBA is the union set of local trigger patterns in DBA, DBA should be implemented before the CBA. The reason can be found in the last paragraph of section 4.1 in the paper.

## Included experiments

| Experiment Name| Dataset| Model |  Number of Clients (`--num_workers`)| Number of Malicious Clients (`--num_mali`)|
|---------------------|-------------------|-------------|---------|-----------|
| Honest Majority Attack Scenario | `NCI1`, `PROTEINS_full`, `TRIANGLES`  | `GCN`, `GAT`, `GraphSAGE`| `5`  | `2` |
| Malicious Majority Attack Scenario | `NCI1`, `PROTEINS_full`, `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `5`  | `3` |
| Impact of the Number of Clients | `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `10`, `20`  | `4`(`6`), `8`(`12`) |
| Impact of the Percentage of Malicious Clients | `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `100` | `5`, `10`, `15`, `20` |
| Defense (`foolsgold` or `flame`) | `NCI1`, `PROTEINS_full`, `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE`| `5` | `2`,`3` |

> Each experiment was repeated 10 times (`--seed`) to get the average result and standard deviation.

## Detailed Usage
### Configuration file
All arguments in the parer are able to set default values in the configuration file in ```./Common/Utils/options.py```

### Some important arguments
> `--dataset` (`default='NCI1', help='name of dataset'`)
> `--datadir` (`default='./Data', help='path to save the downloaded dataset'`)
> `--config` (`default='./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json', help='path of config file which defines the GNN neural network'`)
> `--num_workers` (`default=5, help='number of clients in total'`)
> `--num_mali` (`default=2, help='number of malicious clients in the backdoor attack'`)
> `--filename` (`default='./Results', help='path to save the experimental results'`)

### Script arguments.
The full script arguments are shown below:

```

usage: <script_name>.py [-h] [--num_workers NUM_WORKERS]
                       	[--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
                       	[--weight_decay WEIGHT_DECAY] [--step_size STEP_SIZE]
                       	[--gamma GAMMA] [--dropout DROPOUT]
                       	[--momentum MOMENTUM] [--defense DEFENSE]
                       	[--dataset DATASET] [--datadir DATADIR]
                       	[--config CONFIG] [--target_label TARGET_LABEL]
                       	[--poisoning_intensity POISONING_INTENSITY]
                       	[--frac_of_avg FRAC_OF_AVG] [--density DENSITY]
                       	[--num_mali NUM_MALI] [--filename FILENAME]
                       	[--epoch_backdoor EPOCH_BACKDOOR] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --num_workers NUM_WORKERS
                        number of clients in total (default: 10)
  --batch_size BATCH_SIZE
                        local batch size (default: 128)
  --epochs EPOCHS       training epochs (default: 1000)
  --lr LR               learning rate (default: 0.0007)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 0.0)
  --step_size STEP_SIZE
                        step size (default: 100)
  --gamma GAMMA         gamma (default: 0.9)
  --dropout DROPOUT     drop out (default: 0.0)
  --momentum MOMENTUM   SGD momentum (default: 0.9)
  --defense DEFENSE     whethere perform a defense, e.g., foolsgold, flame
                        (default: None)
  --dataset DATASET     name of dataset (default: NCI1)
  --datadir DATADIR     path to save the dataset (default: ./Data)
  --config CONFIG       Please give a config.json file with model and training
                        details (default: None)
  --target_label TARGET_LABEL
                        target label of the poisoned dataset (default: 0)
  --poisoning_intensity POISONING_INTENSITY
                        frac of training dataset to be injected trigger
                        (default: 0.2)
  --frac_of_avg FRAC_OF_AVG
                        frac of avg nodes to be injected the trigger (default:
                        0.2)
  --density DENSITY     density of the edge in the generated trigger (default:
                        0.8)
  --num_mali NUM_MALI   number of malicious clients (default: 3)
  --filename FILENAME   path of output file(save results) (default: )
  --epoch_backdoor EPOCH_BACKDOOR
                        from which epoch the malicious clients start backdoor
                        attack (default: 0)
  --seed SEED           0-9 (default: 0)
```
