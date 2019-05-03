import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mru import MRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    Encoder for generator, which feeds output into the decoder
    """
    def _encoder_block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            MRU(in_channels, out_channels, self.image_channels, **self.mru_kwargs),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),  # Halve height and width
        )
    
    def __init__(self, num_classes, init_out_channels=64, image_channels=3, 
                 init_image_size=64, image_pool=nn.AvgPool2d(2, stride=2), **kwargs):
        """
        Initialize encoder, consisting of several encoder blocks
        
        Args:
            num_classes: Number of possible classes
            init_out_channels: The number of channels the output of the first encoder block should output
            image_channels: Number of channels for the input images
            init_image_size: The initial size of the images fed into the MRUs
            image_pool: Pooling function to halve size of images fed into MRUs
            **kwargs: Keyword arguments to pass into MRU layer
                norm: Normalization function to use after convolution (without num_features specified)
                mask_norm: True if norm function should also be applied to masks (mi and ni)
                activation: Activation function, f, to use
                sn: True if spectral normalization is used for convolution weights (mainly for discriminator)
        """
        super(Encoder, self).__init__()
        self.image_channels = image_channels
        self.mru_kwargs = kwargs
        out1 = init_out_channels
        out2 = out1 * 2
        out4 = out2 * 2
        out8 = out4 * 2
        
        self.layer1 = self._encoder_block(1, out1)
        self.layer2 = self._encoder_block(out1, out2)
        self.layer3 = self._encoder_block(out2, out4)
        self.layer4 = self._encoder_block(out4, out8)
        self.label_embeds = nn.Embedding(num_classes, init_image_size ** 2)
        self.init_image_size = init_image_size
        self.image_pool = image_pool
    
    def forward(self, labels, images):
        """
        Implements forward pass of encoder
        
        Args:
            labels: Tensor of integer labels for each image of size (batch_size, 1)
            images: Tensor of images of size (batch_size, image_channels, initial_image_size, initial_image_size)
            
        Returns:
            out: Output of encoder of size (batch_size, init_out_channels * 8, height / 8, width / 8)
        """
        ## Convert labels from (batch_size, 1) to size (batch_size, init_image_size, init_image_size)
        embeds = self.label_embeds(labels)
        embeds = embeds.view(-1, init_image_size, init_image_size)
        
        layer1_out = self.layer1(embeds, images)
        images = self.image_pool(layer1_out)
        layer2_out = self.layer2(layer1_out, images)
        images = self.image_pool(layer2_out)
        layer3_out = self.layer3(layer2_out, images)
        images = self.image_pool(layer3_out)
        layer4_out = self.layer4(layer3_out, images)
        return layer4_out
        

# class sketchGAN_G(nn.Moduel):
#     def __init__(self, mru, layer_dims, sampling_dims):
#         # layer_dims is a list of out_channels for each layer
#         # sampling_dims is a list of output shape of each layer (y)
#         super(sketchGAN_G, self).__init__()
#         self.in_channels = None # TODO :: set to initial channel
#         # encoding layers
#         self.enc_layer1 = self.make_layer(mru, layer_dims[0], interpoloate=sampling_dims[0])
#         self.enc_layer2 = self.make_layer(mru, layer_dims[1], interpoloate=sampling_dims[1])
#         self.enc_layer3 = self.make_layer(mru, layer_dims[2], interpoloate=sampling_dims[2])
#         self.enc_layer4 = self.make_layer(mru, layer_dims[3], interpoloate=sampling_dims[3])
#         # decoding layers
#         self.dec_layer1 = self.make_layer(mru, layer_dims[4], interpoloate=sampling_dims[4])
#         self.dec_layer2 = self.make_layer(mru, layer_dims[5], interpoloate=sampling_dims[5])
#         self.dec_layer3 = self.make_layer(mru, layer_dims[6], interpoloate=sampling_dims[6])
#         self.dec_layer4 = self.make_layer(mru, layer_dims[7], interpoloate=sampling_dims[7])
    
#     def make_layer(self, mru, out_channels, activation=nn.ReLU, stride=1, interpoloate=None):
#         layer = mru(self.in_channels, out_channels, activation=activation, stride=stride, interpoloate=interpoloate)
#         self.in_channels = out_channels
#         return nn.Sequential(*layer)

#     def forward(self, x):
#         out1 = self.enc_layer1(x)
#         out2 = self.enc_layer2(out1)
#         out3 = self.enc_layer3(out2)
#         out4 = self.enc_layer4(out3)
#         out5 = self.dec_layer1(out4)
#         # out5 = torch.cat((out5, out4), axis=1) 
#         # Not sure this is needed. This skip-connection is not shown in figure diagram
#         out6 = self.dec_layer2(out5)
#         out6 = torch.cat((out6, out3), axis=1)
#         out7 = self.dec_layer3(out6)
#         out7 = torch.cat((out7, out2), axis=1)
#         out = self.dec_layer4(out7)
#         out = torch.cat((out, out1), axis=1)
#         return out

# model = sketchGAN_G(MRU, layer_dims, sampling_dims).to(device)