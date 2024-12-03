import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.nn import Linear, Dropout, BatchNorm1d
#from tensorlayerx.dataflow import GINConv, SAGEConv
#from torch_geometric.nn import GINConv, SAGEConv



class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, epsilon=0, learn_eps=False):
        super(GINConv, self).__init__()
        self.epsilon = nn.Parameter([epsilon]) if learn_eps else epsilon
        self.mlp = nn.Sequential([
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        ])

    def forward(self, x, edge_index):
        row, col = edge_index
        out = tlx.ops.segment_sum(x[col], row)
        out = (1 + self.epsilon) * x + out
        return self.mlp(out)




class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr="mean"):
        super(SAGEConv, self).__init__()
        self.aggr = aggr
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        
        if self.aggr == "mean":
            out = tlx.ops.segment_mean(x[col], row)
        elif self.aggr == "sum":
            out = tlx.ops.segment_sum(x[col], row)
        elif self.aggr == "max":
            out = tlx.ops.segment_max(x[col], row)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggr}")
        
        out = self.lin_l(x) + self.lin_r(out)
        return tlx.ops.relu(out)

class MLP_discriminator(nn.Module):
    def __init__(self, args):
        super(MLP_discriminator, self).__init__()
        self.args = args
        self.lin = Linear(out_features=1, in_features=args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None, mask_node=None):
        h = self.lin(h)
        return tlx.tlx.sigmoid(h)


class MLP_encoder(nn.Module):
    def __init__(self, args):
        super(MLP_encoder, self).__init__()
        self.args = args
        self.lin = Linear(out_features=args.hidden, in_features=args.num_features)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index=None, mask_node=None):
        h = self.lin(x)
        return h


class GCN_encoder_scatter(nn.Module):
    def __init__(self, args):
        super(GCN_encoder_scatter, self).__init__()
        self.args = args
        self.lin = Linear(out_features=args.hidden, in_features=args.num_features, b_init=None)
        self.bias = tlx.nn.Parameter(shape=(args.hidden,))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.assign(tlx.zeros_like(self.bias))

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = self.propagate(h, edge_index) + self.bias
        return h

    def propagate(self, x, edge_index):
        row, col = edge_index
        deg = tlx.unsorted_segment_sum(tlx.ones_like(row, dtype=tlx.float32), row, tlx.reduce_max(row) + 1)
        deg_inv_sqrt = tlx.pow(deg, -0.5)
        deg_inv_sqrt = tlx.where(deg_inv_sqrt == float('inf'), tlx.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = tlx.unsorted_segment_sum(norm[:, None] * x[col], row, tlx.reduce_max(row) + 1)
        return out


class GCN_encoder_spmm(nn.Module):
    def __init__(self, args):
        super(GCN_encoder_spmm, self).__init__()
        self.args = args
        self.lin = Linear(out_features=args.hidden, in_features=args.num_features, b_init=None)
        self.bias = tlx.nn.Parameter(shape=(args.hidden,))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.assign(tlx.zeros_like(self.bias))

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.lin(x)
        h = tlx.spmm(adj_norm_sp, h) + self.bias
        return h


class GIN_encoder(nn.Module):
    def __init__(self, args):
        super(GIN_encoder, self).__init__()
        self.args = args
        self.mlp = nn.Sequential(
            Linear(out_features=args.hidden, in_features=args.num_features),
            BatchNorm1d(num_features=args.hidden),
        )
        self.conv = GINConv(self.mlp)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        h = self.conv(x, edge_index)
        return h


class SAGE_encoder(nn.Module):
    def __init__(self, args):
        super(SAGE_encoder, self).__init__()
        self.args = args
        self.conv1 = SAGEConv(out_features=args.hidden, in_features=args.num_features, aggr='mean')
        self.transition = nn.Sequential(
            nn.ReLU(),
            BatchNorm1d(num_features=args.hidden),
            Dropout(keep_prob=1-args.dropout)
        )
        self.conv2 = SAGEConv(out_features=args.hidden, in_features=args.hidden, aggr='mean')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, adj_norm_sp):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        h = x
        return h


class MLP_classifier(nn.Module):
    def __init__(self, args):
        super(MLP_classifier, self).__init__()
        self.args = args
        self.lin = Linear(out_features=args.num_classes, in_features=args.hidden)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, h, edge_index=None):
        h = self.lin(h)
        return h
