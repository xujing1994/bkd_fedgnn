import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # federated arguments
    parser.add_argument('--num_workers', type=int, default=10, help="number of clients in total")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--lr', type=float, default=7e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay")
    parser.add_argument('--step_size', type=int, default=100, help="step size")
    parser.add_argument('--gamma', type=float, default=0.7, help="gamma")
    parser.add_argument('--dropout', type=float, default=0.0, help="drop out")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")

    # argument for backdoor attack in GNN model
    parser.add_argument('--dataset', type=str, default="NCI1", help='name of dataset')
    parser.add_argument('--datadir', type=str, default="./Data", help='path to save the dataset')
    parser.add_argument('--config', help="Please give a config.json file with model and training details")
    parser.add_argument('--target_label', type=int, default=0, help='target label of the poisoned dataset')
    parser.add_argument('--poisoning_intensity', type=float, default=0.2, help='frac of training dataset to be injected trigger')
    parser.add_argument('--frac_of_avg', type=float, default=0.2, help='frac of avg nodes to be injected the trigger')
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger')
    parser.add_argument('--num_mali', type=int, default=3, help="number of malicious clients")
    parser.add_argument('--filename', type = str, default = "", help='path of output file(save results)')
    parser.add_argument('--epoch_backdoor', type=int, default=0, help='from which epoch the malicious clients start backdoor attack')
    parser.add_argument('--seed', type=int, default=0, help='0-9')
    parser.add_argument('--defense', type=str, default='None', help='whethere perform a defense, e.g., foolsgold')
    parser.add_argument('--robustLR_threshold', type=int, default=0, 
                        help="break ties when votes sum to 0")
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="To use cuda, set to a specific GPU ID.")
    args = parser.parse_args()
    return args
