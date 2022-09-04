"""
    Utility file to select GraphNN model as
    selected by the user
"""

from GNN_common.nets.TUs_graph_classification.gated_gcn_net import GatedGCNNet
from GNN_common.nets.TUs_graph_classification.gcn_net import GCNNet
from GNN_common.nets.TUs_graph_classification.gat_net import GATNet
from GNN_common.nets.TUs_graph_classification.graphsage_net import GraphSageNet
from GNN_common.nets.TUs_graph_classification.gin_net import GINNet
from GNN_common.nets.TUs_graph_classification.mo_net import MoNet as MoNet_
from GNN_common.nets.TUs_graph_classification.mlp_net import MLPNet
from GNN_common.nets.TUs_graph_classification.ring_gnn_net import RingGNNNet
from GNN_common.nets.TUs_graph_classification.three_wl_gnn_net import ThreeWLGNNNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def MoNet(net_params):
    return MoNet_(net_params)


def MLP(net_params):
    return MLPNet(net_params)


def RingGNN(net_params):
    return RingGNNNet(net_params)


def ThreeWLGNN(net_params):
    return ThreeWLGNNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet_,
        'MLP': MLP,
        'RingGNN': RingGNN,
        '3WLGNN': ThreeWLGNN
    }

    return models[MODEL_NAME](net_params)
