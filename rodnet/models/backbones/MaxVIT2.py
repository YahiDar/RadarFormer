from typing import Tuple, Union, Sequence

import torch
import torch.nn as nn
from maxvit.maxvit import MaxViTBlock
from monai.networks.blocks import  UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
import math
from monai.utils import  optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")





class RadarStackedHourglass(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num = 1, use_mse_loss=False,
                 patch_size = 8, receptive_field = [[3,3,3,3],[3,3,3,3]], hidden_size = 516, num_layers = 12,
                win_size=16, out_head =1):
        super(RadarStackedHourglass, self).__init__()


        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([UNETR(
                                                        in_channels=in_channels,
                                                        out_channels=n_class,
                                                        img_size=(win_size, 128, 128),
                                                        feature_size=patch_size,
                                                        hidden_size=hidden_size,
                                                        out_head = out_head,
                                                        num_layers=num_layers,
                                                        receptive_field = receptive_field,
                                                        pos_embed='perceptron',
                                                    ),
                                                 ]))        

        self.hourglass = nn.ModuleList(self.hourglass)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss



    def forward(self, x):
        confmap = self.hourglass[0][0](x)
        if not self.use_mse_loss:
                confmap = self.sigmoid(confmap)
        return confmap







class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, window_size, kernel_size = 3,  stride = 1):
        super(ConvLayer, self).__init__()

        self.spatial = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = kernel_size, stride = stride, padding = kernel_size//2)

    def forward(self, x):
        x = self.spatial(x)
        return x




class SingleConv(nn.Module):

    def __init__(self, kernal_size, in_channels, out_channels, window_size, stride = 1):
        super(SingleConv, self).__init__()

        self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels,
                                window_size = window_size, kernel_size = kernal_size, stride = stride)
                                
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)
    

    def forward(self, x):
        x = self.bn1(self.block1(x))
        
        return x


class PostConv(nn.Module):

    def __init__(self, kernal_size, in_channels, window_size, out_channels, stride =1):
        super(PostConv, self).__init__()

        b1channel = in_channels + (out_channels - in_channels)//2
        self.block1 = ConvLayer(in_channels = in_channels, out_channels = b1channel,
                                window_size = window_size, kernel_size = kernal_size, stride = stride)
                                
        self.block2 = ConvLayer(in_channels = b1channel, out_channels = out_channels,
                                window_size = window_size, kernel_size = kernal_size, stride = stride)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features = out_channels)


    def forward(self, x):
        x = self.relu(self.block1(x))
        x = self.bn1(self.block2(x))
        x = self.relu(x)
        return x

class ConvRes(nn.Module):

    def __init__(self, kernal_size, in_channels, out_channels, window_size, stride =1 ):
        super(ConvRes, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels != self.out_channels:
            self.ch_conv = ConvLayer(in_channels = in_channels, out_channels = out_channels,
            window_size = window_size, kernel_size = kernal_size, stride = stride)
            self.block1 = ConvLayer(in_channels = out_channels, out_channels = out_channels//2,
            window_size = window_size, kernel_size = kernal_size, stride = stride)
        else:
            self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels//2,
                                window_size = window_size, kernel_size = kernal_size, stride = stride)
        self.block2 = ConvLayer(in_channels = out_channels//2, out_channels = out_channels,
                                window_size = window_size, kernel_size = kernal_size, stride = stride)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features = out_channels//2)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels)



        
    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.ch_conv(x)        
        res = x
        out = self.relu(self.bn1(self.block1(x)))
        out = self.relu(self.bn2(self.block2(out)))
        


        return (out + res)

class Interpolate(nn.Module):
    def __init__(self, size, mode = 'bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class UpsampleStep(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernal_size = None,
                window_size = 32, 
                up_size = None, 
                patch_size = None, 
                steps = 1, 
                uptype = 'transpose'):

        super(UpsampleStep, self).__init__()

        self.uptype = uptype
        self.up_size = up_size
        self.patch_size = patch_size
        self.steps = steps

        if self.uptype == 'transpose':
            self.layer = UnetrPrUpBlock(
            spatial_dims = 2,
            in_channels = in_channels,
            out_channels = out_channels,
            num_layer = steps - 1,
            kernel_size = kernal_size,
            stride = 1,
            upsample_kernel_size = 2,
            norm_name = 'batch',
            conv_block = False,
            res_block = False,
        )
        
        elif self.uptype == 'interpolate':
            self.layer = Interpolate(size = up_size, mode = 'bilinear')

        elif self.uptype == 'mlp':
            
            self.feat_size = (
            self.up_size[0] // self.patch_size[0],
            self.up_size[1] // self.patch_size[1],
                             )
            self.rearrange = Rearrange('b (w d) (p1 p2 c) -> b c (w p1) (d p2)',
                                    p1 = self.patch_size[0], p2 = self.patch_size[1], 
                                    w = self.feat_size[0] ,d = self.feat_size[1])

            self.layer = nn.Linear(in_channels,(patch_size[0]*patch_size[1]*out_channels))

    def forward(self, x):
        if self.uptype == 'mlp':
            x = self.layer(x)
            x = self.rearrange(x)
        elif (self.uptype == 'transpose_inception' and self.steps > 1):
            for i in range(self.steps):
                x = self.layer[i](x)

        else:
            x = self.layer(x)

        return x



class UNETR(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 8,
        hidden_size: int = 516,
        num_layers: int = 12,
        out_head: int = 1,
        receptive_field: list = [[3,3,3,3],[3,3,3,3]],
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
    ) -> None:


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
        self.rf = receptive_field
        self.img_size = (img_size[1],img_size[2])
        self.num_layers = num_layers//4
        self.patch_size = (feature_size, feature_size)
        self.out_channels = out_channels 
        self.feat_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.out_head = out_head
        self.num_samples = int(math.log(int(feature_size),2))

        self.hidden_size = hidden_size
        self.classification = False
        self.vit1 = MaxViTBlock(in_channels = in_channels*3,
            out_channels= in_channels*3,
            downscale = False,
            num_heads = (in_channels*3)//2,
            grid_window_size = (self.patch_size[0], self.patch_size[0]),
            mlp_ratio = hidden_size//((in_channels*3)//2),
            act_layer = nn.GELU,
            norm_layer = nn.BatchNorm2d,
            norm_layer_transformer = nn.LayerNorm
    )

        self.vit2 = MaxViTBlock(in_channels = feature_size * 2,
            out_channels= feature_size * 2,
            downscale = False,
            num_heads = (feature_size*2)//2,
            grid_window_size = (self.patch_size[0], self.patch_size[0]),
            mlp_ratio = hidden_size//((feature_size*2)//2),
            act_layer = nn.GELU,
            norm_layer = nn.BatchNorm2d,
            norm_layer_transformer = nn.LayerNorm
    )

        self.vit3 = MaxViTBlock(in_channels = feature_size * 4,
            out_channels= feature_size * 4,
            downscale = False,
            num_heads = (feature_size*4)//2,
            grid_window_size = (self.patch_size[0], self.patch_size[0]),
            mlp_ratio = hidden_size//((feature_size * 4)//2),
            act_layer = nn.GELU,
            norm_layer = nn.BatchNorm2d,
            norm_layer_transformer = nn.LayerNorm
    )

        self.vit4 = MaxViTBlock(in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
            out_channels= feature_size * 8 if (feature_size * 8) <=64 else 64,
            downscale = False,
            num_heads = (feature_size * 8 if (feature_size * 8) <=64 else 64)//2,
            grid_window_size = (self.patch_size[0], self.patch_size[0]),
            mlp_ratio = hidden_size//((feature_size * 8 if (feature_size * 8) <=64 else 64)//2),
            act_layer = nn.GELU,
            norm_layer = nn.BatchNorm2d,
            norm_layer_transformer = nn.LayerNorm
    )
       
        k_3d = 1+img_size[0]//7 
        self.encoder11 = nn.Conv3d(
              kernel_size=(k_3d, 5, 5),
              in_channels = in_channels,
              out_channels = in_channels*2,
              padding=(0, 2, 2),
              stride = (2,1,1))

        self.encoder12 = nn.Conv3d(
              kernel_size=(k_3d, 5, 5),
              in_channels = in_channels*2,
              out_channels = in_channels*2,
              padding=(0, 2, 2),
              stride = (2,1,1))

        self.encoder13 = nn.Conv3d(
              kernel_size=(k_3d, 5, 5),
              in_channels = in_channels*2,
              out_channels = in_channels*3,
              padding=(0, 2, 2),
              stride = (1,1,1))
        self.bn11 = nn.BatchNorm3d(in_channels*2)
        self.bn12 = nn.BatchNorm3d(in_channels*2)
        self.bn13 = nn.BatchNorm3d(in_channels*3)
        self.relu = nn.ReLU()

        self.res1_decoder3 = SingleConv(kernal_size = self.rf[0][0], in_channels = in_channels*3,
                                        window_size = img_size[0], out_channels= in_channels*3, stride = 1)

        self.res1_decoder4 = SingleConv(kernal_size = self.rf[0][0], in_channels = in_channels*3,
                                        window_size = img_size[0], out_channels= feature_size*2, stride = 1)

        self.res2_decoder3 = SingleConv(kernal_size = self.rf[0][1], in_channels = feature_size * 2,
                                        window_size = img_size[0], out_channels= feature_size * 2, stride = 1)
        self.res2_decoder4 = SingleConv(kernal_size = self.rf[0][1], in_channels = feature_size *2,
                                        window_size = img_size[0], out_channels= feature_size*4, stride = 1)



        self.res3_decoder3 = SingleConv(kernal_size = self.rf[0][2], in_channels = feature_size * 4,
                                        window_size = img_size[0], out_channels= feature_size * 4, stride = 1)
        self.res3_decoder4 = SingleConv(kernal_size = self.rf[0][2], in_channels = feature_size *4,
                                        window_size = img_size[0], out_channels= feature_size * 8 if (feature_size * 8) <=64 else 64, stride = 1)

          


        self.final_decoder1 = SingleConv(in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                                        out_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                                        kernal_size= self.rf[0][3],
                                        stride = 1,
                                        window_size = self.feat_size[0])
        self.final_decoder2 = SingleConv(in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                                        out_channels = in_channels * 3,
                                        kernal_size= self.rf[0][3],
                                        stride = 1,
                                        window_size = self.feat_size[0])

        self.merge1 = nn.ConvTranspose3d(
              kernel_size = (k_3d, 5, 5),
              in_channels = in_channels*3 ,
              out_channels = in_channels*2,
              padding = (0,2,2),
              stride = (2,1,1))
        self.merge2 = nn.ConvTranspose3d(
              kernel_size = (k_3d if (img_size[0]) == 16 else (k_3d+1), 5, 5),
              in_channels = in_channels*2,
              out_channels = in_channels*2,
              padding = (0,2,2),
              stride = (2,1,1))
        self.merge3 = nn.ConvTranspose3d(
              kernel_size = (k_3d+1, 5, 5),
              in_channels = in_channels*2,
              out_channels = in_channels,
              padding = (0,2,2),
              stride = (2,1,1))
        
        #self.bn20 = nn.BatchNorm2d(in_channels*3)
        self.bn21 = nn.BatchNorm3d(in_channels*2)
        self.bn22 = nn.BatchNorm3d(in_channels*2)
        self.bn23 = nn.BatchNorm3d(in_channels)
        
        self.out = nn.Conv3d(in_channels=in_channels, out_channels=self.out_channels, kernel_size = self.out_head,stride=1)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    # def get_memory_free_MiB(self, gpu_index):
    #     pynvml.nvmlInit()
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    #     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #     return mem_info.free // 1024 ** 2

    def forward(self, x_input):
        x_in1 = self.relu(self.bn11(self.encoder11(x_input)))
        x_in2 = self.relu(self.bn12(self.encoder12(x_in1)))
        x_in3 = self.relu(self.bn13(self.encoder13(x_in2)))
        x_in = torch.squeeze(x_in3, dim = 2)
        x = self.vit1(x_in)
        x1 = self.res1_decoder3(x)
        x1 = self.res1_decoder4(x1 + x_in)
        x = self.vit2(x1)
        x2 = self.res2_decoder3(x)
        x2 = self.res2_decoder4(x2 + x1)
        x = self.vit3(x2)
        x3 = self.res3_decoder3(x)
        x3 = self.res3_decoder4(x2 + x3)
        x = self.vit4(x3)
        x4 = self.final_decoder1(x)
        x4 = self.final_decoder2(x4+x3)

        x4 = torch.unsqueeze(x4,dim=2)
        
        x4 = self.relu(self.bn21(self.merge1(x4 + x_in3)))
        x4 = self.relu(self.bn22(self.merge2(x4 + x_in2)))
        x4 = self.relu(self.bn23(self.merge3(x4 + x_in1)))
        out = self.out(x4)
        return out


