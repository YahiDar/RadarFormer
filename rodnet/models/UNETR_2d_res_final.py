import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

from .backbones.UNETR_2d_res_final import RadarStackedHourglass
from .modules.mnet import MNet

class UNETR_2d_res_final(nn.Module):
    def __init__(self, 
                in_channels, 
                n_class, 
                stacked_num=1, 
                mnet_cfg=None, 
                dcn=True,
                out_head = 1,
                win_size = 16,
                patch_size = 8, 
                norm_layer = 'batch',
                hidden_size = 516, 
                receptive_field = [[3,3,3,3],[3,3,3,3]],
                mlp_dim = 3072,
                num_layers = 12, 
                num_heads = 12):
        super(UNETR_2d_res_final, self).__init__()
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
                                                out_head = out_head, num_heads = num_heads)
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
    testModel = UNETR_2d_res_final().cuda()
    x = torch.zeros((1, 2, 16, 128, 128)).cuda()
    testModel(x)
