# More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks

This repo contais the source code used in our paper. It consists of 3 main
python scripts used for different tasks. In particular, `clean_fedgnn.py`
trains federated GNNs without any backdoors, `dis_bkd_fedgnn.py` trains
backdoored federated GNNs under the distributed backdoor attack (DBA) setup,
and `cen_bkd_fedgnn.py` trains backdoored federated GNNs under the centralized
backdoor atttack (CBA) setup.

## Installation
We tested our code with python 3.6.8 and python 3.9.2. So as long as there
is python >= 3.6 installed everything can be easily tested through a virtual
environment which can be created and activated in the following way:
```
$ python -m venv env
$ . env/bin/activate
```

Then install the dependencies based on ```requirements.txt```.

```
$ python -m pip install -r requirements.txt
```

## Dataset
The dataset can be specified by setting `--dataset` with dataset name, such as ``NCI1``, ``PROTEINS_full``, ``TRIANGLES``. The dataset name can be those on [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/).

## Test run
### 1. Train a clean Federated GNN model
```
python clean_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 0 --filename ./Results/Clean
```
> Note: The results will be saved in the folder `./Results/Clean` and among the saved files, the file named `GCN_NCI1_5_0_0.20_0.20_0.80_global_test.txt` contains the test accuracy of the global model, which is used to calculate the clean accuracy drop, as presented in Section 5.3 in the paper.

### 2. Test distributed backdoor attack in Federated GNNs
```
python dis_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2 --filename ./Results/DBA
```
### 3. Test centralized backdoor attack in Federated GNNs
```
python cen_bkd_fedgnn.py --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2 --filename ./Results/CBA
```
> Note: For each script of backdoor attack in Federated GNNs, we can get the train loss, train accuracy, test loss, test accuracy, attack success rate with global trigger, attack success rate with each local trigger for each client, and the test accuracy, attack success rate with global trigger, attack success rate with each local trigger for the global model, for each epoch, as follows:
```
epoch: 0
Client 0, loss 0.6251, train acc 0.628, test loss 0.6968, test acc 0.492
Client 0 with global trigger: 1.000
Client 0 with local trigger 0: 1.000
Client 0 with local trigger 1: 1.000
Client 1, loss 0.5336, train acc 0.714, test loss 0.6824, test acc 0.562
Client 1 with global trigger: 1.000
Client 1 with local trigger 0: 1.000
Client 1 with local trigger 1: 1.000
Client 2, loss 0.8846, train acc 0.489, test loss 0.6891, test acc 0.585
Client 2 with global trigger: 1.000
Client 2 with local trigger 0: 1.000
Client 2 with local trigger 1: 0.901
Client 3, loss 0.6916, train acc 0.543, test loss 0.6999, test acc 0.467
Client 3 with global trigger: 0.758
Client 3 with local trigger 0: 0.714
Client 3 with local trigger 1: 0.033
Client 4, loss 0.6709, train acc 0.604, test loss 0.7165, test acc 0.467
Client 4 with global trigger: 0.099
Client 4 with local trigger 0: 0.110
Client 4 with local trigger 1: 0.011
Global Test Acc: 0.579
Global model with global trigger: 1.000
Global model with local trigger 0: 1.000
Global model with local trigger 1: 0.736
```
> Note: The results of DBA or CBA will be saved in the folder `./Results/DBA` or `./Results/CBA`. The attack results of the global model is saved in the file named `GCN_NCI1_5_2_0.20_0.20_0.80_global_attack.txt`. Specifically, the first column is the attack success rate with the global trigger and the other columns are the attack success rate with local triggers, which are used to draw Figure 3, 4, 5, 7, 11, 12 in the paper.

### 4. Test defense against backdoor attack in Federated GNNs

Here one defense can be tested against the backdoor attack in Federated GNNs, by setting value of `--defense` to be `foolsgold`.
This defense is implemented following the algorithms in the paper: [Mitigating Sybils in Federated Learning Poisoning](https://arxiv.org/abs/1808.04866).

Example:
```
python dis_bkd_fedgnn.py --defense foolsgold --dataset NCI1 --config ./GNN_common/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json --num_workers 5 --num_mali 2 --filename ./Results/DBA_foolsgold
```
> Note: For each script of backdoor attack in Federated GNNs with defense, the backdoor attack results with defense will be obtained, as well as the weights on every client in FoolsGold (i.e., alpha) which are reported to explain the ineffectiveness of FoolsGold, as shown in Appendix B in the paper. The output of the script of FoolsGold defense can be seen as follows:

```
epoch: 0
Client 0, loss 0.6288, train acc 0.631, test loss 0.6974, test acc 0.492
Client 0 with global trigger: 1.000
Client 0 with local trigger 0: 1.000
Client 0 with local trigger 1: 1.000
Client 1, loss 0.5270, train acc 0.714, test loss 0.6821, test acc 0.562
Client 1 with global trigger: 1.000
Client 1 with local trigger 0: 1.000
Client 1 with local trigger 1: 1.000
Client 2, loss 0.9303, train acc 0.489, test loss 0.6883, test acc 0.600
Client 2 with global trigger: 1.000
Client 2 with local trigger 0: 1.000
Client 2 with local trigger 1: 1.000
Client 3, loss 0.7069, train acc 0.504, test loss 0.6985, test acc 0.467
Client 3 with global trigger: 0.484
Client 3 with local trigger 0: 0.516
Client 3 with local trigger 1: 0.615
Client 4, loss 0.6686, train acc 0.607, test loss 0.7146, test acc 0.467
Client 4 with global trigger: 0.022
Client 4 with local trigger 0: 0.044
Client 4 with local trigger 1: 0.011
alpha:
[1.         1.         1.         0.54983845 0.54983845]
Global Test Acc: 0.533
Global model with global trigger: 1.000
Global model with local trigger 0: 1.000
Global model with local trigger 1: 1.000
```
> Note: The results of DBA (CBA) with FoolsGold defense will be saved in the folder `./Results/{}_{}.format(attack, defense)`, e.g., `./Results/DBA_foolsgold`. Still, the file named `GCN_NCI1_5_2_0.20_0.20_0.80_global_attack.txt` contains the attack results of the global model, which represents the backdoor attack results on defense and is used to draw Figure 9, 10, 13, 14 in the paper. In addition, for FoolsGold, the value of alpha will be saved in a file `GCN_NCI1_5_2_0.20_0.20_0.80_alpha.txt` in the folder `./Results/alpha/DBA` or `./Results/alpha/CBA`. In this file, each column is the aggregation weight of each client.

> Note: The experimental results won't be saved without value for `--filename`.
> 
> Note: In order to make sure trigger pattern in CBA is the union set of local trigger patterns in DBA, DBA should be implemented before the CBA. The reason can be found in the last paragraph of Section 4.1 in the paper.

## Included experiments

| Experiment Name| Dataset| Model |  Number of Clients (`--num_workers`)| Number of Malicious Clients (`--num_mali`)|
|---------------------|-------------------|-------------|---------|-----------|
| Honest Majority Attack Scenario | `NCI1`, `PROTEINS_full`, `TRIANGLES`  | `GCN`, `GAT`, `GraphSAGE`| `5`  | `2` |
| Malicious Majority Attack Scenario | `NCI1`, `PROTEINS_full`, `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `5`  | `3` |
| Impact of the Number of Clients | `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `10`, `20`  | `4`(`6`), `8`(`12`) |
| Impact of the Percentage of Malicious Clients | `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE` | `100` | `5`, `10`, `15`, `20` |
| Defense (`foolsgold`) | `NCI1`, `PROTEINS_full`, `TRIANGLES` |`GCN`, `GAT`, `GraphSAGE`| `5` | `2`,`3` |

> Each experiment was repeated 10 times with a different seed each time
> (`--seed {1-10}`) to get the average result and standard deviation.

## Detailed Usage
There are many arguments that control the operation of our scripts. These
arguments are contained in ```./Common/Utils/options.py``` and shown below:

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
  --defense DEFENSE     whethere perform a defense, e.g., foolsgold
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
