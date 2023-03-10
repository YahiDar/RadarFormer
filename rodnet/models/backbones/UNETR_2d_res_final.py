from typing import Tuple, Union, Sequence

import torch
import torch.nn as nn

from monai.networks.blocks import  UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
import math
from monai.utils import  optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")





class RadarStackedHourglass(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num = 1, conv_op=None, use_mse_loss=False,
                 patch_size = 8, norm_layer = 'batch', receptive_field = [[3,3,3,3],[3,3,3,3]], hidden_size = 516,
                 mlp_dim = 3072, num_layers = 12, num_heads = 12, win_size=16, out_head =1):
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
                                                        out_head = out_head,
                                                        num_layers=num_layers,
                                                        num_heads=num_heads,
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
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, window_size, kernel_size = 3,  stride = 1):
        super(ConvLayer, self).__init__()

        self.spatial = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = kernel_size, stride = stride, padding = kernel_size//2)

    def forward(self, x):
        # B, C, WS, H, W = x.shape
        # x = torch.reshape(x.permute(0,2,1,3,4),[B*WS,C,H,W])
        x = self.spatial(x)
        # x = torch.reshape(x,[B,WS,C,H,W]).permute(0,2,1,3,4)
        # x = self.temporal(x)
        return x



class InceptionLayerTrans(nn.Module):


    def __init__(self, kernal_size, in_channel, out_channel, window_size, stride = 1):
        super(InceptionLayerTrans, self).__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(out_channel)

        self.branch2a = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, 
                                        kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.branch2b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                    kernel_size = kernal_size, window_size = window_size*2, stride = stride,)

        self.branch3a = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, 
                                        kernel_size=(2, 2), stride=(2, 2), bias=False)
            
        self.branch3b = ConvLayer(in_channels= out_channel, out_channels=out_channel,
                                    kernel_size = kernal_size, window_size = window_size*2, stride = stride)

        self.merge = ConvLayer(in_channels= out_channel*2, out_channels=out_channel,
                                    kernel_size = kernal_size, window_size = window_size*2, stride = 1)

    def forward(self, x):

        branch2 = self.branch2a(x)
        branch2 = self.relu(self.branch2b(branch2))

        branch3 = self.branch3a(x)
        branch3 = self.relu(self.branch3b(branch3))
        
        out = self.relu(self.bn1(self.merge(torch.cat((branch2, branch3), 1))))

        return out




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

        if self.uptype == 'transpose_inception':
            if steps == 1:
                self.layer = InceptionLayerTrans(kernal_size = kernal_size, in_channel = in_channels,
                                        window_size = window_size, out_channel = out_channels, stride=1)
            else:
                self.layer = nn.ModuleList(
                                    [
                    InceptionLayerTrans(kernal_size = kernal_size, in_channel = in_channels,
                                        window_size = window_size, out_channel = in_channels, stride=1)
                                        for i in range(steps-1)
                                    ]
                                    )
                self.layer.append(InceptionLayerTrans(kernal_size = kernal_size, in_channel = in_channels,
                                        window_size = window_size, out_channel = out_channels, stride=1))
                
        
        elif self.uptype == 'transpose':
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
        out_head: int = 1,
        num_heads: int = 12,
        receptive_field: list = [[3,3,3,3],[3,3,3,3]],
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
        #self.rearrange_2d = Rearrange("b c w H W -> (b w) c H W")

        #self.reverse_rearrange = Rearrange("(b w) c H W -> b c w H W", w = img_size[0])

        self.hidden_size = hidden_size
        self.classification = False
        self.vit1 = ViT(
            in_channels=in_channels*3,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
            
        )

        self.vit2 = ViT(
            in_channels= feature_size * 2,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.vit3 = ViT(
            in_channels=feature_size * 4,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        self.vit4 = ViT(
            in_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            spatial_dims = 2,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        # self.encoder1 = UnetrBasicBlock(
        #     spatial_dims=3,
        #     in_channels=in_channels,
        #     out_channels=feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
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

        

        self.res1_decoder1 = UpsampleStep(
                in_channels = hidden_size,
                out_channels = feature_size*2,
                kernal_size = 3,
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = 1, 
                uptype = 'transpose_inception')
                
        self.res1_decoder2 = UpsampleStep(
                in_channels=feature_size * 2,
                out_channels=feature_size * 2,
                kernal_size = self.rf[0][0],
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = self.num_samples - 1, 
                uptype = 'transpose')

        self.res1_decoder3 = SingleConv(kernal_size = self.rf[0][0], in_channels = feature_size *2,
                                        window_size = img_size[0], out_channels= feature_size*2, stride = 1)
        self.res1_decoder4 = SingleConv(kernal_size = self.rf[0][0], in_channels = in_channels*3,
                                        window_size = img_size[0], out_channels= feature_size*2, stride = 1)
        # self.encoder2 = UnetrPrUpBlock(
        #     spatial_dims=3,
        #     in_channels=hidden_size,
        #     out_channels=feature_size * 2,
        #     num_layer=2,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )
        self.res2_decoder1 = UpsampleStep(
                in_channels = hidden_size,
                out_channels = feature_size*4,
                kernal_size = 3,
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = 1, 
                uptype = 'transpose_inception')
                
        self.res2_decoder2 = UpsampleStep(
                in_channels=feature_size * 4,
                out_channels=feature_size * 4,
                kernal_size = self.rf[0][1],
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = self.num_samples - 1, 
                uptype = 'transpose')


        self.res2_decoder3 = SingleConv(kernal_size = self.rf[0][1], in_channels = feature_size *4,
                                        window_size = img_size[0], out_channels= feature_size*4, stride = 1)
        self.res2_decoder4 = SingleConv(kernal_size = self.rf[0][1], in_channels = feature_size *2,
                                        window_size = img_size[0], out_channels= feature_size*4, stride = 1)


       
        # self.encoder3 = UnetrPrUpBlock(
        #     spatial_dims=3,
        #     in_channels=hidden_size,
        #     out_channels=feature_size * 4,
        #     num_layer=1,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )
        self.res3_decoder1 = UpsampleStep(
                in_channels = hidden_size,
                out_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                kernal_size = 3,
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = 1, 
                uptype = 'transpose_inception')
                
        self.res3_decoder2 = UpsampleStep(
                in_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
                out_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
                kernal_size = self.rf[0][2],
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = self.num_samples - 1, 
                uptype = 'transpose')


        self.res3_decoder3 = SingleConv(kernal_size = self.rf[0][2], in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                                        window_size = img_size[0], out_channels= feature_size * 8 if (feature_size * 8) <=64 else 64, stride = 1)
        self.res3_decoder4 = SingleConv(kernal_size = self.rf[0][2], in_channels = feature_size *4,
                                        window_size = img_size[0], out_channels= feature_size * 8 if (feature_size * 8) <=64 else 64, stride = 1)
        # self.encoder4 = UnetrPrUpBlock(
        #     spatial_dims=3,
        #     in_channels=hidden_size,
        #     out_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
        #     num_layer=0,
        #     kernel_size=3,
        #     stride=1,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     conv_block=conv_block,
        #     res_block=res_block,
        # )
        self.res4_decoder1 = UpsampleStep(
                in_channels = hidden_size,
                out_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                kernal_size = 3,
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = 1, 
                uptype = 'transpose_inception')
                
        self.res4_decoder2 = UpsampleStep(
                in_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
                out_channels=feature_size * 8 if (feature_size * 8) <=64 else 64,
                kernal_size = self.rf[0][3],
                window_size = 32, 
                up_size = (128,128), 
                patch_size = (8,8), 
                steps = self.num_samples - 1, 
                uptype = 'transpose')
          


        self.final_decoder = SingleConv(in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
                                        out_channels = in_channels * 3,
                                        kernal_size= 3,
                                        stride = 1,
                                        window_size = self.feat_size[0])
        # self.res43_sum = PostConv(kernal_size = self.rf[1][0], in_channels = feature_size * 8 if (feature_size * 8) <=64 else 64,
        #                          window_size = img_size[0], out_channels = feature_size * 8 if (feature_size * 8) <=64 else 64, stride = 1)
        # self.res42_sum = PostConv(kernal_size = self.rf[1][1], in_channels = feature_size * 12 if (feature_size * 8) <=64 else 128,
        #                          window_size = img_size[0], out_channels = feature_size * 4, stride = 1)
        # self.res41_sum = PostConv(kernal_size = self.rf[1][2], in_channels = feature_size * 6,
        #                          window_size = img_size[0], out_channels = in_channels , stride = 1)
        # self.res40_sum = PostConv(kernal_size = self.rf[1][3], in_channels = feature_size * 2,
        #                          window_size = img_size[0], out_channels = feature_size , stride = 1)
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
        #x_in = self.rearrange_2d(x_in)
        x = self.vit1(x_in)
        x1 = self.proj_feat(x, self.hidden_size, self.feat_size)
        x1 = self.res1_decoder2(self.res1_decoder1(x1))
        x1 += self.res1_decoder4(x_in)
        x1 = self.res1_decoder3(x1)
        x = self.vit2(x1)
        x2 = self.proj_feat(x, self.hidden_size, self.feat_size)
        x2 = self.res2_decoder2(self.res2_decoder1(x2))
        x2 += self.res2_decoder4(x1)
        x2 = self.res2_decoder3(x2)
        x = self.vit3(x2)
        x3 = self.proj_feat(x, self.hidden_size, self.feat_size)
        x3 = self.res3_decoder2(self.res3_decoder1(x3))
        x3 += self.res3_decoder4(x2)
        x3 = self.res3_decoder3(x3)
        x = self.vit4(x3)
        x4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        x4 = self.res4_decoder2(self.res4_decoder1(x4))
        x4 = self.final_decoder(x4+x3)
        #x4 += x_in
        #x4 = self.bn20(x4)
        # x4 = self.res43_sum(x4)
        # x4 = torch.cat((x4,x2), dim=1)
        # x4 = self.res42_sum(x4)
        # x4 = torch.cat((x4,x1), dim=1)
        # x4 = self.res41_sum(x4)
        ## up to 3d
        #x4 = self.reverse_rearrange(x4)
        x4 = torch.unsqueeze(x4,dim=2)
        # x_in = self.reverse_rearrange(x_in)
        #merge with original
        
        x4 = self.relu(self.bn21(self.merge1(x4 + x_in3)))
        x4 = self.relu(self.bn22(self.merge2(x4 + x_in2)))
        x4 = self.relu(self.bn23(self.merge3(x4 + x_in1)))
        out = self.out(x4)
        return out


