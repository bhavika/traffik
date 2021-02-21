import torch
from torch import nn
from torch_geometric.nn import ChebConv, LayerNorm, SAGEConv, SGConv
from torch.nn import functional as F


class KipfBlock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, k=8, p=0.5, apply_batch_norm=False):
        super(KipfBlock, self).__init__()
        self.conv1 = ChebConv(n_input, n_hidden, k=k)
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.apply_batch_norm = apply_batch_norm
        if self.apply_batch_norm:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.apply_batch_norm:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))
        return x


class SageBlock(torch.nn.Module):
    def __init__(self, n_input, n_hidden, apply_batch_norm=False):
        super(SageBlock, self).__init__()
        self.conv1 = SAGEConv(n_input, n_hidden)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.apply_batch_norm = apply_batch_norm
        if self.apply_batch_norm:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.apply_batch_norm:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))
        return x


class SgBlock(torch.nn.Module):
    def __init__(self, n_input, n_hidden, k=3, apply_batch_norm=False):
        super(SgBlock, self).__init__()
        self.conv1 = SGConv(n_input, n_hidden, k, cached=False)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.apply_batch_norm = apply_batch_norm
        if self.apply_batch_norm:
            self.bn = LayerNorm(n_hidden)

    def forward(self, x, edge_index):
        if self.apply_batch_norm:
            x = F.elu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))
        return x


class GraphEnsembleNet(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        nh=38,
        k=6,
        k_mix=2,
        inout_skipconn=True,
        depth=3,
        p=0.5,
        apply_batch_norm=False,
    ):
        super(GraphEnsembleNet, self).__init__()
        self.inout_skipconn = inout_skipconn
        self.depth = depth

        self.KipfBlock_list = nn.ModuleList()
        self.Sage_list = nn.ModuleList()
        self.Sg_list = nn.ModuleList()

        if isinstance(nh, list):
            # if you give every layer a different number of channels
            # you need one number of channels for every layer!
            assert len(nh) == depth

        else:
            channels = nh
            nh = []
            for i in range(depth):
                nh.append(channels)

        for i in range(depth):
            if i == 0:
                self.KipfBlock_list.append(
                    KipfBlock(
                        n_input=num_features,
                        n_hidden=nh[0],
                        k=k,
                        p=p,
                        apply_batch_norm=apply_batch_norm,
                    )
                )
                self.Sage_list.append(SageBlock(num_features, nh[0]))
                self.Sg_list.append(SgBlock(num_features, nh[0], k=5))
            else:
                self.KipfBlock_list.append(
                    KipfBlock(
                        n_input=nh[i - 1],
                        n_hidden=nh[i],
                        k=k,
                        p=p,
                        apply_batch_norm=apply_batch_norm,
                    )
                )
                self.Sage_list.append(SageBlock(nh[i - 1], nh[0]))
                self.Sg_list.append(SgBlock(nh[i - 1], nh[0], k=5))

        if inout_skipconn:
            self.conv_mix = ChebConv(nh[-1] + num_features, num_classes, k=k_mix)
            self.sage_mix = SageBlock(nh[-1] + num_features, num_classes)
            self.sg_mix = SgBlock(nh[-1] + num_features, num_classes, k=5)
        else:
            self.conv_mix = ChebConv(nh[-1], num_classes, k=k_mix)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.depth):
            x = (
                self.Kipfblock_list[i](x, edge_index)
                + self.Sage_list[i](x, edge_index)
                + self.Sg_list[i](x, edge_index)
            ) / 3

        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            x = (
                self.conv_mix(x, edge_index)
                + self.sage_mix(x, edge_index)
                + self.sg_mix(x, edge_index)
            ) / 3
        else:
            x = self.conv_mix(x, edge_index)

        return x
