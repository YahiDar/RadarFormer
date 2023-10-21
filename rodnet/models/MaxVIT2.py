import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

from .backbones.MaxVIT2 import RadarStackedHourglass
from .modules.mnet import MNet




class MaxVIT2(nn.Module):
    def __init__(self, 
                in_channels, 
                n_class, 
                stacked_num=1, 
                mnet_cfg=None, 
                dcn=False,
                out_head = 1,
                win_size = 16,
                patch_size = 8, 
                hidden_size = 516, 
                receptive_field = [[3,3,3,3],[3,3,3,3]],
                num_layers = 12):
        super(MaxVIT2, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
                                                num_layers = num_layers, receptive_field = receptive_field,
                                                out_head = out_head)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
                                                num_layers = num_layers, receptive_field = receptive_field,
                                                out_head = out_head)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out


if __name__ == '__main__':
    testModel = MaxVIT2().cuda()
    x = torch.zeros((1, 2, 16, 128, 128)).cuda()
    testModel(x)
