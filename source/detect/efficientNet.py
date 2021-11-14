import warnings
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from collections import OrderedDict
import torch

class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers):
      
        orig_return_layers = return_layers.copy()
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, md in model.named_children():
            if name != "_blocks":
                layers[name] = md
                if name in return_layers:
                    del return_layers[name]
                if not return_layers:
                    break
            else:
                for name_c, i in md.named_children():
                    
                    layers[name_c] = i
                    if name_c in return_layers:
                        del return_layers[name_c]
                    if not return_layers:
                        break

        super().__init__(layers)
        self.return_layers = orig_return_layers


    def forward(self, x):
        out = OrderedDict()

        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out

class BackboneWithFPN_B7(nn.Module):
    def __init__(
        self,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks,
    ):
        
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x