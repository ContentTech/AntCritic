from torch import nn
import torch_geometric.nn as graph_nn

class GraphComponent(nn.Module):
    # Checked √
    def __init__(self, name, layer_num, dropout, gnn_config):
        super().__init__()
        self.layer_num = layer_num
        self.name = name
        self.conv_layers, self.ffn_layers = self._construct_deep_layers(name, layer_num, dropout, gnn_config)

    def _construct_deep_layers(self, name, layer_num, dropout, gnn_config):
        conv_layers = []
        ffn_layers = []
        for layer_idx in range(layer_num):
            conv = getattr(graph_nn, name + "Conv")(**gnn_config)
            if "heads" in gnn_config:
                out_channels = gnn_config["out_channels"] * gnn_config["heads"]
            else:
                out_channels = gnn_config["out_channels"]
            norm = nn.LayerNorm(out_channels)
            act = nn.GELU()
            conv_layers.append(graph_nn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout))

            ffn_layers.append(nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.Linear(out_channels, out_channels * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels * 4, out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        return nn.ModuleList(conv_layers), nn.ModuleList(ffn_layers)

    def forward(self, node_feat, edge_index, layer_idx):
        conv_feat = self.conv_layers[layer_idx](node_feat, edge_index)
        final_feat = node_feat + self.ffn_layers[layer_idx](conv_feat)
        return final_feat
