import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    #parser.add_argument('--model', type=str, default="GIN", help="model name")
    #parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--num_workers', type=int, default=10, help="number of users: N")
    parser.add_argument('--E', type=int, default=1, help="the number of local updates: E")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay")
    parser.add_argument('--step_size', type=int, default=100, help="step size")
    parser.add_argument('--gamma', type=float, default=0.9, help="gamma")
    parser.add_argument('--dropout', type=float, default=0.0, help="drop out")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--id', type=int, default=0, help='client id (default: 1)')
    parser.add_argument('--path', type=str, default='./Data/MNIST/', help="dataset")
    parser.add_argument('--defense', type=str, default='None', help='whethere there is a defense, e.g., foolsgold, flame')
    # argument for GNN model training
    #parser.add_argument('--datadir', type=str, default="./TUDataset")
    parser.add_argument('--hidden_channels', type=int, default=64, help='number of hidden units (default: 64)')
    #parser.add_argument('--num_layers', type=int, default=3, help='number of layers (default: 3)')
    parser.add_argument('--dataset', type=str, default="NCI1", help='name of dataset (default: MUTAG)')
    #parser.add_argument('--sparse', action='store_false', help='the graph is sparse or dense')
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--target_label', type=int, default=0, help='target label')
    parser.add_argument('--poisoning_intensity', type=float, default=0.2, help='frac of training dataset to be injected trigger')
    parser.add_argument('--frac_of_avg', type=float, default=0.2, help='frac of avg nodes to be injected the trigger')
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger')
    parser.add_argument('--num_mali', type=int, default=3, help="number of malicious clients")
    parser.add_argument('--filename', type = str, default = "", help='output file')
    parser.add_argument('--epoch_backdoor', type=int, default=0, help='from which epoch the malicious clients start backdoor attack')
    #parser.add_argument('--num_round', type=int, default=100, help='number of rounds')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    #parser.add_argument('--same_local_trigger', action='store_true', help='whether use the same local trigger for all malicious clients in DBA')

    args = parser.parse_args()
    return args
