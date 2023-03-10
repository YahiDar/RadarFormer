from typing import Tuple, Union, Sequence

import torch
import torch.nn as nn

import pynvml
from monai.networks.blocks import  UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
import math
from monai.utils import  optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")





class RadarStackedHourglass(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num = 1, conv_op=None, use_mse_loss=False,
                 patch_size = 8, norm_layer = 'batch', receptive_field = [3,3,3,3], hidden_size = 516,
                 mlp_dim = 3072, num_layers = 12, num_heads = 12, win_size=16,channels_features = (1,2,3,4)):
        super(RadarStackedHourglass, self).__init__()


        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([UNETR(
                                                        in_channels=in_channels,
                                                        out_channels=n_class,
                                                        img_size=(win_size, 128, 128),
                                                        feature_size=patch_size,
                                                        hidden_size=hidden_size,
                                                        mlp_dim=mlp_dim,
                                                        num_layers=num_layers,
                                                        num_heads=num_heads,
                                                        channels_features = channels_features,
                                                        receptive_field = receptive_field,
                                                        pos_embed='perceptron',
                                                        norm_name=norm_layer,
                                                        conv_block=True,
                                                        res_block=True,
                                                        dropout_rate=0.0,
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

__all__ = ["ViT"]


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore


    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, window_size, kernel_size = 3,  stride = 1, padding = None, spatial_dims = 3):
        super(ConvLayer, self).__init__()
        if padding == None:
            padding = kernel_size // 2
        
        if spatial_dims == 3:
            self.spatial = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = kernel_size, stride = stride, padding = padding)
        elif spatial_dims == 2:
            self.spatial = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = kernel_size, stride = stride, padding = padding)
    def forward(self, x):
        # B, C, WS, H, W = x.shape
        # x = torch.reshape(x.permute(0,2,1,3,4),[B*WS,C,H,W])
        x = self.spatial(x)
        # x = torch.reshape(x,[B,WS,C,H,W]).permute(0,2,1,3,4)
        # x = self.temporal(x)
        return x



class InceptionLayerTrans(nn.Module):


    def __init__(self, kernal_size, in_channel, out_channel, window_size, stride = 1, padding = None, spatial_dims = 3):
        super(InceptionLayerTrans, self).__init__()

        self.relu = nn.ReLU()
        if spatial_dims == 3:
            self.bn1 = nn.BatchNorm3d(out_channel)

            self.branch2a = nn.ConvTranspose3d(in_channels = in_channel, out_channels = out_channel, 
                                            kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
            self.branch2b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = stride, 
                                        padding = padding, spatial_dims = spatial_dims)

            self.branch3a = nn.ConvTranspose3d(in_channels = in_channel, out_channels = out_channel, 
                                            kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
                
            self.branch3b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = stride, 
                                        padding = padding, spatial_dims = spatial_dims)

            self.merge = ConvLayer(in_channels= out_channel*2, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = 1,
                                        padding = padding, spatial_dims = spatial_dims)
        elif spatial_dims ==2:
            self.bn1 = nn.BatchNorm2d(out_channel)

            self.branch2a = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, 
                                            kernel_size=(2, 2), stride=(2, 2), bias=False)
            self.branch2b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = stride,
                                        padding = padding, spatial_dims = spatial_dims)

            self.branch3a = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, 
                                            kernel_size=(2, 2), stride=(2, 2), bias=False)
                
            self.branch3b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = stride,
                                        padding = padding, spatial_dims = spatial_dims)

            self.merge = ConvLayer(in_channels= out_channel*2, out_channels=out_channel,
                                        kernel_size = kernal_size, window_size = window_size*2, stride = 1,
                                        padding = padding, spatial_dims = spatial_dims)

    def forward(self, x):

        branch2 = self.branch2a(x)
        branch2 = self.relu(self.branch2b(branch2))

        branch3 = self.branch3a(x)
        branch3 = self.relu(self.branch3b(branch3))
        
        out = self.relu(self.bn1(self.merge(torch.cat((branch2, branch3), 1))))

        return out




class SingleConv(nn.Module):

    def __init__(self, kernal_size, in_channels, out_channels, window_size, stride = 1, padding = None, spatial_dims = 3):
        super(SingleConv, self).__init__()
        if spatial_dims == 3:
            self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, 
                                    padding = padding, spatial_dims = spatial_dims)
                                    
            self.bn1 = nn.BatchNorm3d(num_features = out_channels)
            
        elif spatial_dims == 2:
            self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels,
                                window_size = window_size, kernel_size = kernal_size, stride = stride,
                                padding = padding, spatial_dims = spatial_dims)
                                
            self.bn1 = nn.BatchNorm2d(num_features = out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.block1(x)))
        
        return x


class PostConv(nn.Module):

    def __init__(self, kernal_size, in_channels, window_size, out_channels, stride =1, padding = None, spatial_dims = 3):
        super(PostConv, self).__init__()

        b1channel = in_channels + (out_channels - in_channels)//2
        self.relu = nn.ReLU()
        if spatial_dims == 3:
            self.block1 = ConvLayer(in_channels = in_channels, out_channels = b1channel,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, 
                                    padding = padding, spatial_dims = spatial_dims)
                                    
            self.block2 = ConvLayer(in_channels = b1channel, out_channels = out_channels,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride,
                                    padding = padding, spatial_dims = spatial_dims)
            self.bn1 = nn.BatchNorm3d(num_features = out_channels)
        elif spatial_dims == 2:
            self.block1 = ConvLayer(in_channels = in_channels, out_channels = b1channel,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, 
                                    padding = padding, spatial_dims = spatial_dims)
                                    
            self.block2 = ConvLayer(in_channels = b1channel, out_channels = out_channels,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, 
                                    padding = padding, spatial_dims = spatial_dims)
            self.bn1 = nn.BatchNorm2d(num_features = out_channels)
            


    def forward(self, x):
        x = self.relu(self.block1(x))
        x = self.bn1(self.block2(x))
        x = self.relu(x)
        return x

class ConvRes(nn.Module):

    def __init__(self, kernal_size, in_channels, out_channels, window_size, stride =1, padding = None, spatial_dims = 3 ):
        super(ConvRes, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        if spatial_dims == 3:
            if self.in_channels != self.out_channels:
                self.ch_conv = ConvLayer(in_channels = in_channels, out_channels = out_channels,
                window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
                self.block1 = ConvLayer(in_channels = out_channels, out_channels = out_channels//2,
                window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            else:
                self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels//2,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            self.block2 = ConvLayer(in_channels = out_channels//2, out_channels = out_channels,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            self.bn1 = nn.BatchNorm3d(num_features = out_channels//2)
            self.bn2 = nn.BatchNorm3d(num_features = out_channels)
        elif spatial_dims == 2:
            if self.in_channels != self.out_channels:
                self.ch_conv = ConvLayer(in_channels = in_channels, out_channels = out_channels,
                window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
                self.block1 = ConvLayer(in_channels = out_channels, out_channels = out_channels//2,
                window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            else:
                self.block1 = ConvLayer(in_channels = in_channels, out_channels = out_channels//2,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            self.block2 = ConvLayer(in_channels = out_channels//2, out_channels = out_channels,
                                    window_size = window_size, kernel_size = kernal_size, stride = stride, padding = padding, spatial_dims = spatial_dims)
            self.bn1 = nn.BatchNorm2d(num_features = out_channels//2)
            self.bn2 = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()




        
    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.ch_conv(x)        
        res = x
        out = self.relu(self.bn1(self.block1(x)))
        out = self.relu(self.bn2(self.block2(out)))
        


        return (out + res)

class Interpolate(nn.Module):
    def __init__(self, size, mode, to_3d = False, in_channels = None, out_channels = None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.to_3d = to_3d
        if self.to_3d == True:
            self.conv = nn.ConvTranspose3d(in_channels = in_channels,
                                           out_channels = out_channels,
                                           kernel_size = (4,1,1),
                                           stride = (2,1,1)
                                           )

        
    def forward(self, x):
        if self.to_3d == True:
            x = torch.unsqueeze(x,dim=2)
            x = self.conv(x)

        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, steps = 0, spatial_dims = 3):
        super(DownSample,self).__init__()
        self.spatial_dims = spatial_dims
        self.steps = steps
        if spatial_dims == 3:

            self.blocks1 = nn.Conv3d(in_channels = in_channels,
                                     out_channels = in_channels,
                                     kernel_size = (3,1,1),
                                     stride=(2,1,1))
            self.blocks2 = nn.Conv3d(in_channels = in_channels,
                                     out_channels = in_channels,
                                     kernel_size = (3,1,1),
                                     stride=(2,1,1))
            self.blocks3 = nn.Conv3d(in_channels = in_channels,
                                     out_channels = in_channels,
                                     kernel_size = (4,1,1),
                                     stride=(4,1,1))
        if steps > 0:
            self.blocks = nn.ModuleList(
                                        [
                        nn.Conv2d(in_channels = in_channels,
                                out_channels = in_channels,
                                kernel_size = kernel_size,
                                stride = 2,
                                padding = kernel_size//2
                        )   for i in range(steps)
                                        ]
                                        )




    def forward(self,x):
        if self.spatial_dims == 3:
            x = self.blocks1(x)
            x = self.blocks2(x)
            x = self.blocks3(x)
            x = torch.squeeze(x,dim=2)
        if self.steps > 0:
            for blk in self.blocks:
                x = blk(x)

        return x



class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 8,
        hidden_size: int = 516,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        channels_features: Tuple = (1,2,3,4),
        num_heads: int = 12,
        receptive_field: list = [3,3,3,3],
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "batch",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and batch norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='batch')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")
        self.rf = receptive_field
        self.size_step = (1,1,2,4)
        self.img_size = (img_size[0]//self.size_step[0],img_size[1]//self.size_step[0],img_size[2]//self.size_step[0])
        self.img_size2 = (self.img_size[1]//self.size_step[1], self.img_size[2]//self.size_step[1])
        self.img_size3 = (self.img_size[1]//self.size_step[2], self.img_size[2]//self.size_step[2])
        self.img_size4 = (self.img_size[1]//self.size_step[3],self.img_size[2]//self.size_step[3])
        self.num_layers = num_layers//4
        self.in_channels = in_channels
        self.patch_size = (feature_size, feature_size, feature_size)
        self.patch_size2 = (feature_size, feature_size)
        self.out_channels = out_channels 
        self.feat_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.feat_size2 = (
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )

        self.num_samples = int(math.log(int(feature_size),2))
        self.cf = channels_features
        self.hidden_size = hidden_size
        self.classification = False
        self.interpolation_mode3d = 'trilinear'
        self.interpolation_mode2d = 'bilinear'


        ##################### INTERPOLATION FUNCTIONS

        self.upsample2_to1 = nn.ModuleList(
                                        [
                                        Interpolate(
                                        size = self.img_size, 
                                         mode = self.interpolation_mode3d,
                                         to_3d = True, 
                                         in_channels = in_channels*self.cf[1], 
                                         out_channels = in_channels*self.cf[0]
                                         ) for i in range(4)
                                        ]
                                        )
        self.upsample3_to1 = nn.ModuleList(
                                        [
                                        Interpolate(
                                        size = self.img_size, 
                                         mode = self.interpolation_mode3d,
                                         to_3d = True, 
                                         in_channels = in_channels*self.cf[2], 
                                         out_channels = in_channels*self.cf[0]
                                         ) for i in range(3)
                                        ]
                                        )

        self.upsample4_to1 = nn.ModuleList(
                                        [
                                        Interpolate(
                                        size = self.img_size, 
                                         mode = self.interpolation_mode3d,
                                         to_3d = True, 
                                         in_channels = in_channels*self.cf[3], 
                                         out_channels = in_channels*self.cf[0]
                                         ) for i in range(2)
                                        ]
                                        )
        self.upsample_to2 = Interpolate(size = self.img_size2, 
                                        to_3d = False,
                                        mode = self.interpolation_mode2d)
        self.upsample_to3 = Interpolate(size = self.img_size3,
                                        to_3d = False,
                                        mode = self.interpolation_mode2d)




        #################### BASIC STAGES BLOCKS #

        #### INPUT
        self.input_stage1 = SingleConv(kernal_size = 1,
                            in_channels = in_channels*self.cf[0],
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            stride = 1)

        self.input_stage2 = SingleConv(kernal_size = 3,
                    in_channels = in_channels*self.cf[0],
                    out_channels = in_channels*self.cf[0],
                    window_size = self.img_size[0],
                    stride = 1) 

        self.input_stage3 = SingleConv(kernal_size = 1,
                            in_channels = in_channels*self.cf[0],
                            out_channels = in_channels *self.cf[1],
                            window_size = self.img_size[0],
                            stride = 1)
    

    ###################### FIRST LEVEL


        self.level1_conv1 = SingleConv(kernal_size = self.rf[0],
                            in_channels = in_channels*self.cf[1],
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 3)
        self.level1_conv2 = SingleConv(kernal_size = self.rf[0],
                            in_channels = in_channels*self.cf[0],
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 3)
        self.level1_conv3 = SingleConv(kernal_size = self.rf[0],
                            in_channels = in_channels*self.cf[0],
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 3)
        self.level1_conv4 = SingleConv(kernal_size = self.rf[0],
                            in_channels = in_channels*self.cf[0],
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 3)

######################## SECOND LEVEL

        # self.level2_conv1 = SingleConv(kernal_size = 3,
        #                     in_channels = in_channels*2,
        #                     out_channels = in_channels*2,
        #                     window_size = self.img_size[0],
        #                     stride = 1)
        self.level2_conv2 = SingleConv(kernal_size = self.rf[1],
                            in_channels = in_channels*self.cf[1],
                            out_channels = in_channels*self.cf[1],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)
        self.level2_conv3 = SingleConv(kernal_size = self.rf[1],
                            in_channels = in_channels*self.cf[1],
                            out_channels = in_channels*self.cf[1],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)
        self.level2_conv4 = SingleConv(kernal_size = self.rf[1],
                            in_channels = in_channels*self.cf[1],
                            out_channels = in_channels*self.cf[1],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)
######################## THIRD LEVEL

        # self.level3_conv1 = SingleConv(kernal_size = 3,
        #                     in_channels = (in_channels + in_channels*2),
        #                     out_channels = in_channels*4,
        #                     window_size = self.img_size[0],
        #                     stride = 1)
        self.level3_conv2 = SingleConv(kernal_size = self.rf[2],
                            in_channels = in_channels*self.cf[2],
                            out_channels = in_channels*self.cf[2],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)
        self.level3_conv3 = SingleConv(kernal_size = self.rf[2],
                            in_channels =  in_channels*self.cf[2],
                            out_channels = in_channels*self.cf[2],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)

######################## Fourth LEVEL

        # self.level4_conv1 = SingleConv(kernal_size = 3,
        #                     in_channels = (in_channels*2 + in_channels + in_channels*4),
        #                     out_channels = in_channels*8,
        #                     window_size = self.img_size[0],
        #                     stride = 1)
        self.level4_conv2 = SingleConv(kernal_size = self.rf[3],
                            in_channels = in_channels*self.cf[3],
                            out_channels = in_channels*self.cf[3],
                            window_size = self.img_size[0],
                            stride = 1, 
                            spatial_dims = 2)


####################### Transformers

################## FIRST LEVEL:

        self.level1_t1 = ViT(
            in_channels=in_channels*self.cf[0],
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 3,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.level1_t2 = ViT(
            in_channels=in_channels*(self.cf[0]*2),
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 3,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.level1_t3 = ViT(
            in_channels=in_channels*(self.cf[0]*3),
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 3,
            classification=self.classification,

        )

        self.project_level1_t1 = nn.Linear(hidden_size,(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]*in_channels*self.cf[0]))
        self.project_level1_t2 = nn.Linear(hidden_size,(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]*in_channels*self.cf[0]))
        self.project_level1_t3 = nn.Linear(hidden_size,(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]*in_channels*self.cf[0]))

        self.rearrange1 = Rearrange('b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)',
                                    p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2], 
                                    h = self.feat_size[0] ,w = self.feat_size[1],d = self.feat_size[2])

################## SECOND LEVEL:

        self.level2_t1 = ViT(
            in_channels=in_channels*self.cf[1],
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.level2_t2 = ViT(
            in_channels=in_channels*(self.cf[0]+self.cf[1]),
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.level2_t3 = ViT(
            in_channels=in_channels*(self.cf[0]+self.cf[1]+self.cf[2]),
            img_size=self.img_size2,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.project_level2_t1 = nn.Linear(hidden_size,(self.patch_size2[0]*self.patch_size2[1]*in_channels*self.cf[1]))
        self.project_level2_t2 = nn.Linear(hidden_size,(self.patch_size2[0]*self.patch_size2[1]*in_channels*self.cf[1]))
        self.project_level2_t3 = nn.Linear(hidden_size,(self.patch_size2[0]*self.patch_size2[1]*in_channels*self.cf[1]))
        self.rearrange2 = Rearrange('b (w d) (p2 p3 c) -> b c (w p2) (d p3)',
                                    p2=self.patch_size2[0], p3=self.patch_size2[1], 
                                    w = self.feat_size2[0],d = self.feat_size2[1])


################## THIRD LEVEL:

        self.level3_t1 = ViT(
            in_channels = in_channels*(self.cf[0]+self.cf[1]),
            img_size=self.img_size3,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )

        self.level3_t2 = ViT(
            in_channels = in_channels*(self.cf[0]+self.cf[1]+self.cf[2]),
            img_size=self.img_size3,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )
        self.project_level3_t1 = nn.Linear(hidden_size,(self.patch_size2[0]*self.patch_size2[1]*in_channels*self.cf[2]))
        self.project_level3_t2 = nn.Linear(hidden_size,(self.patch_size2[0]*self.patch_size2[1]*in_channels*self.cf[2]))
        self.rearrange3 = Rearrange('b (w d) (p2 p3 c) -> b c (w p2) (d p3)',
                                    p2=self.patch_size2[0], p3=self.patch_size2[1], 
                                    w = self.feat_size2[0]//2,d = self.feat_size[1]//2)


################## FOURTH LEVLE:

        self.level4_t1 = ViT(
            in_channels = in_channels*(self.cf[0]+self.cf[1]+self.cf[2]),
            img_size=self.img_size4,
            patch_size=self.patch_size2,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )

        self.project_level4_t1 = nn.Linear(hidden_size,((self.patch_size[0])*(self.patch_size[1])*in_channels*(self.cf[3])))
        self.rearrange4 = Rearrange('b (w d) (p2 p3 c) -> b c (w p2) (d p3)',
                                    p2=self.patch_size2[0], p3=self.patch_size2[1], 
                                    w = self.feat_size2[0]//4,d = self.feat_size2[1]//4)


####################### DOWNSAMPLING

#### LEVEL 1
        self.level1_down21 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[0], steps = 0, spatial_dims = 3)
        self.level1_down22 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 0, spatial_dims = 3)
        self.level1_down23 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 0, spatial_dims = 3)
        self.level1_down24 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 0, spatial_dims = 3)

        self.level1_down31 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 1, spatial_dims = 3)
        self.level1_down32 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 1, spatial_dims = 3)
        self.level1_down33 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = self.rf[0], steps = 1, spatial_dims = 3)

        self.level1_down41 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = 3, steps = 2, spatial_dims = 3)
        self.level1_down42 = DownSample(in_channels = in_channels*self.cf[0], kernel_size = 3, steps = 2, spatial_dims = 3)

### LEVEL 2

        self.level2_down31 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[1], steps = 1, spatial_dims = 2)
        self.level2_down32 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[1], steps = 1, spatial_dims = 2)
        self.level2_down33 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[1], steps = 1, spatial_dims = 2)

        self.level2_down41 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[1], steps = 2, spatial_dims = 2)
        self.level2_down42 = DownSample(in_channels = in_channels*self.cf[1], kernel_size = self.rf[1], steps = 2, spatial_dims = 2)

#### LEVEL 3

        self.level3_down41 = DownSample(in_channels = in_channels*self.cf[2], kernel_size = self.rf[2], steps = 1, spatial_dims = 2)
        self.level3_down42 = DownSample(in_channels = in_channels*self.cf[2], kernel_size = self.rf[2], steps = 1, spatial_dims = 2)



#### OUT CONVS
        self.out1 = SingleConv(kernal_size = self.rf[0],
                            in_channels = in_channels*(self.cf[0]*4),
                            out_channels = in_channels*self.cf[0],
                            window_size = self.img_size[0],
                            spatial_dims = 3,
                            stride = 1)
        self.out2 = SingleConv(kernal_size = self.rf[1],
                            in_channels = in_channels*(self.cf[0]+self.cf[1]+self.cf[2]+self.cf[3]),
                            out_channels = in_channels*self.cf[1],
                            window_size = self.img_size[0],
                            spatial_dims = 2,
                            stride = 1)
        self.out3 = SingleConv(kernal_size = self.rf[2],
                            in_channels = in_channels*(self.cf[0]+self.cf[1]+self.cf[2]+self.cf[3]),
                            out_channels = in_channels*self.cf[2],
                            window_size = self.img_size[0],
                            spatial_dims = 2,
                            stride = 1)

        self.out4 = SingleConv(kernal_size = self.rf[3],
                            in_channels = in_channels*(self.cf[0]+self.cf[1]+self.cf[2]+self.cf[3]),
                            out_channels = in_channels*self.cf[3],
                            window_size = self.img_size[0],
                            spatial_dims = 2,
                            stride = 1)

        self.out = UnetOutBlock(spatial_dims=3, in_channels=(in_channels*4*self.cf[0]), out_channels=self.out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def get_memory_free_MiB(self, gpu_index):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.free // 1024 ** 2

    def forward(self, x):
            
            x = self.input_stage3(self.input_stage2(self.input_stage1(x)))
            x2 = self.level1_down21(x)
            x = self.level1_conv1(x)
            x = self.level1_t1(x)
            x = self.project_level1_t1(x)
            x = self.rearrange1(x)
            x2 = self.level2_t1(x2)
            x2 = self.project_level2_t1(x2)
            x2 = self.rearrange2(x2)

            x3 = torch.cat(
                (self.level1_down31(x),
                self.level2_down31(x2)),
                dim = 1
            )

            x2_hold = self.upsample2_to1[0](x2)
            x2 = self.level2_conv2(x2)
            x2 = torch.cat(
                (self.level1_down22(x),
                 x2),
                dim = 1
            )
            x = self.level1_conv2(x)
            
            x = torch.cat(
                (x,
                x2_hold),
                dim=1
                )
            del x2_hold
            x = self.level1_t2(x)
            x = self.project_level1_t2(x)
            x = self.rearrange1(x)


            x2 = self.level2_t2(x2)
            x2 = self.project_level2_t2(x2)
            x2 = self.rearrange2(x2)


            x3 = self.level3_t1(x3)
            x3 = self.project_level3_t1(x3)
            x3 = self.rearrange3(x3)

            x4 = torch.cat(
                (self.level1_down41(x),
                self.level2_down41(x2),
                self.level3_down41(x3)),
                dim = 1
            )

            x3_hold = self.upsample_to2(x3) 
            x3 = self.level3_conv2(x3)

            x3 = torch.cat(
                (self.level1_down32(x),
                self.level2_down32(x2),
                x3),
                dim = 1
            )

            
            x2_hold = self.upsample2_to1[1](x2)
            x2 = self.level2_conv3(x2)
            x2 = torch.cat(
                (self.level1_down23(x),
                 x2,
                 x3_hold),
                dim = 1
            )


            x3_hold = self.upsample3_to1[0](x3_hold)
            x = self.level1_conv3(x)

            x = torch.cat(
                (x,
                 x2_hold,
                 x3_hold),
                dim = 1
            )
            del x3_hold
            del x2_hold

            x = self.level1_t3(x)
            x = self.project_level1_t3(x)
            x = self.rearrange1(x)
            x2 = self.level2_t3(x2)
            x2 = self.project_level2_t3(x2)
            x2 = self.rearrange2(x2)
            x3 = self.level3_t2(x3)
            x3 = self.project_level3_t2(x3)
            x3 = self.rearrange3(x3)
            x4 = self.level4_t1(x4)
            x4 = self.project_level4_t1(x4)
            x4 = self.rearrange4(x4)
            
            x4_hold = self.upsample_to3(x4)
            x4 = self.level4_conv2(x4)
            x4 = torch.cat(
                (self.level1_down42(x),
                self.level2_down42(x2),
                self.level3_down42(x3),
                x4),
                dim = 1
            )
            
            x4 = self.out4(x4)
            
            x3_hold = self.upsample_to2(x3)
            x3 = self.level3_conv3(x3)

            x3 = torch.cat(
                (self.level1_down33(x),
                self.level2_down33(x2),
                x3,
                x4_hold),
                dim = 1
            )
            x3 = self.out3(x3)
            
            x4_hold = self.upsample_to2(x4_hold)
            x2_hold = self.upsample2_to1[2](x2)
            x2 = self.level2_conv4(x2)



            x2 = torch.cat(
                (self.level1_down23(x),
                 x2,
                 x3_hold,
                 x4_hold),
                dim = 1
            )
            x2 = self.out2(x2)
            

            x4_hold = self.upsample4_to1[0](x4_hold)
        
            x3_hold = self.upsample3_to1[1](x3_hold)
            x = self.level1_conv4(x)
            x = torch.cat(
                (x,
                 x2_hold,
                 x3_hold,
                 x4_hold),
                dim = 1
            )
            
            del x4_hold
            del x3_hold
            del x2_hold


            x = self.out1(x)


            x2 = self.upsample2_to1[3](x2)
            x3 = self.upsample3_to1[2](x3)
            x4 = self.upsample4_to1[1](x4)
            x = torch.cat(
                (x,x2,x3,x4),
                dim = 1
            )
            

            x = self.out(x)
            
            return x



