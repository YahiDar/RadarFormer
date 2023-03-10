import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

from .backbones.HRFormer2d import RadarStackedHourglass
from .modules.mnet import MNet



class HRFormer2d(nn.Module):
    def __init__(self, 
                in_channels, 
                n_class, 
                stacked_num=1, 
                mnet_cfg=None, 
                dcn=True,
                win_size = 16,
                patch_size = 8, 
                norm_layer = 'batch',
                hidden_size = 516, 
                channels_features = (1,2,3,4),
                receptive_field = [3,3,3,3],
                mlp_dim = 3072,
                num_layers = 12, 
                num_heads = 12):
        super(HRFormer2d, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
                                                win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
                                                num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, 
                                                num_heads = num_heads,channels_features = channels_features)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out


if __name__ == '__main__':
    testModel = HRFormer2d().cuda()
    x = torch.zeros((1, 2, 16, 128, 128)).cuda()
    testModel(x)
