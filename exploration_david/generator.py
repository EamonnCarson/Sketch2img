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
    def _encoder_block(self, in_channels, out_channels=None, pool=True):
        """
        Encoder block, which consists of MRU unit + downsampling
        
        Args:
            in_channels: Number of input channels fed into MRU
            out_channels: Number of output channels from MRU. Set to None to default to in_channels * 2
            pool: Set True to perform downsampling after MRU unit
        """
        if out_channels is None:
            out_channels = in_channels * 2
        ml = nn.ModuleList([MRU(in_channels, out_channels, self.image_channels, **self.mru_kwargs)])
        if pool:
            ml.append(nn.Conv2d(out_channels, out_channels, 2, stride=2))  # Halve height and width
        return ml
    
    def __init__(self, num_classes, init_in_channels, init_out_channels=64, image_channels=3, 
                 init_image_size=64, image_pool=nn.AvgPool2d(2, stride=2), **kwargs):
        """
        Initialize encoder, consisting of several encoder blocks
        
        Args:
            num_classes: Number of possible classes
            init_in_channels: The number of channels the initial input feature maps have
            init_out_channels: The number of channels the first encoder block should output
            image_channels: Number of channels for the input images
            init_image_size: The initial size of the images fed into the MRUs
            image_pool: Pooling function to halve size of images fed into MRUs
            **kwargs: Keyword arguments to pass into MRU layer
                norm: Normalization function to use after convolution (without num_features specified)
                mask_norm: True if norm function should also be applied to masks (mi and ni)
                activation: Activation function, f, to use
                mask_activation: Activation function to use on masks mi and ni
                sn: True if spectral normalization is used for convolution weights (mainly for discriminator)
        """
        super(Encoder, self).__init__()
        self.image_channels = image_channels
        self.init_image_size = init_image_size
        self.image_pool = image_pool
        self.mru_kwargs = kwargs
        
        self.layers = nn.ModuleList([
            self._encoder_block(init_in_channels, out_channels=init_out_channels), 
            self._encoder_block(init_out_channels), 
            self._encoder_block(init_out_channels * 2), 
            self._encoder_block(init_out_channels * 4, pool=False)
        ])
        
    def _forward_encoder_block(self, block, input_maps, images):
        """
        Runs the forward pass through a single encoder block.
        
        Args:
            layer: Encoder block to run forward pass through
            input_maps: Tensor of input feature maps of size (batch_size, in_channels, height, width)
            images: Tensor of images of size (batch_size, out_channels, height, width)
        """
        mru_out = block[0]((input_maps, images))
        out = mru_out
        for layer in block[1:]:
            out = layer(out)
        return mru_out, out
    
    def forward(self, input_tuple):
        """
        Implements forward pass of encoder
        
        Args:
            input_maps: Tensor of input feature maps of size (batch_size, init_in_channels, init_image_size, init_image_size)
            images: Tensor of images of size (batch_size, image_channels, init_image_size, init_image_size)
            
        Returns:
            out: Array of the outputs of the encoder blocks. 
            The final encoder output has size (batch_size, init_out_channels * 8, init_image_size / 8, init_image_size / 8)
        """
        input_maps, images = input_tuple
        mru_outputs = []
        out = input_maps
        for layer in self.layers:
            mru_out, out = self._forward_encoder_block(layer, out, images)
            mru_outputs.append(mru_out)
            images = self.image_pool(images)
        return mru_outputs
        

class Decoder(nn.Module):
    def _decoder_block(self, in_channels, out_channels=None, deconv_halve=True, upsample=True, **kwargs):
        """
        Decoder block, which consists of MRU unit + upsampling
        
        Args:
            in_channels: Number of input channels fed into MRU
            out_channels: Number of output channels from MRU. Set to None to default to in_channels / 2
            deconv_halve: Number of output channels of deconvolution. Set True for out_channels / 2, else out_channels
            upsample: Set True to use a deconvolution layer after the MRU unit
            **kwargs: Keyword arguments to pass into MRU layer
        """
        if out_channels is None:
            out_channels = int(in_channels / 2)
        ml = nn.ModuleList([MRU(in_channels, out_channels, self.image_channels, **self.mru_kwargs)])
        if upsample:
            if deconv_halve:
                deconv_channels = int(out_channels / 2)
            else:
                deconv_channels = out_channels
            ml.append(nn.ConvTranspose2d(out_channels, deconv_channels, 2, stride=2))
        return nn.Sequential(*ml)
    
    def __init__(self, init_out_channels=64, image_channels=3, init_image_size=64, 
                 image_pool=nn.AvgPool2d(2, stride=2), **kwargs):
        """
        Initialize decoder, consisting of several decoder blocks
        
        Args:
            init_out_channels: The number of channels the first encoder block outputted
            image_channels: Number of channels for the input images
            init_image_size: The initial size of the images fed into the MRUs
            image_pool: Pooling function to halve size of images fed into MRUs
            **kwargs: Keyword arguments to pass into MRU layer
                norm: Normalization function to use after convolution (without num_features specified)
                mask_norm: True if norm function should also be applied to masks (mi and ni)
                activation: Activation function, f, to use
                mask_activation: Activation function to use on masks mi and ni
                sn: True if spectral normalization is used for convolution weights (mainly for discriminator)
        """
        super(Decoder, self).__init__()
        self.image_channels = image_channels
        self.image_pool = image_pool
        self.mru_kwargs = kwargs
        
        self.layer1 = self._decoder_block(init_out_channels * 16)
        self.layer2 = self._decoder_block(init_out_channels * 8)
        self.layer3 = self._decoder_block(init_out_channels * 4)
        self.layer4 = self._decoder_block(init_out_channels * 2, upsample=False)
        self.layer5 = nn.Sequential(nn.Conv2d(init_out_channels, 3, 3, padding=1), nn.Tanh())  # Should there be norm here?
        
    def forward(self, input_tuple):
        """
        Implements forward pass of decoder
        
        Args:
            encoder_output: Output from forward pass of Encoder
            images: Tensor of images of size (batch_size, image_channels, init_image_size, init_image_size)
            
        Returns:
            out: Output from the decoder, with size (batch_size, 3, init_image_size, init_image_size)
            noise: The Gaussian noise concatenated at the bottleneck
        """
        encoder_output, images = input_tuple
        encoder_output.reverse()
        # Create list of resized images from (init_image_size, init_image_size) to (init_image_size / 8, init_image_size / 8)
        images_resized = [images]
        smaller_images = images
        for i in range(3):
            smaller_images = self.image_pool(smaller_images)
            images_resized.append(smaller_images)
        images_resized.reverse()
        
        # Concat Gaussian noise to output of encoder
        noise = torch.randn(encoder_output[0].size())
        out = torch.cat((encoder_output[0], noise), dim=1)
        
        out = self.layer1((out, images_resized[0]))
        out = torch.cat((encoder_output[1], out), dim=1)
        out = self.layer2((out, images_resized[1]))
        out = torch.cat((encoder_output[2], out), dim=1)
        out = self.layer3((out, images_resized[2]))
        out = torch.cat((encoder_output[3], out), dim=1)
        out = self.layer4((out, images_resized[3]))
        out = self.layer5(out)
        return out, noise
    

class Generator(nn.Module):
    def __init__(self, num_classes, init_out_channels=64, image_channels=3, 
                 init_image_size=64, image_pool=nn.AvgPool2d(2, stride=2), **kwargs):
        """
        Initialize generator, consisting of encoder and decoder
        
        Args:
            num_classes: Number of possible classes
            init_out_channels: The number of channels the first encoder block should output
            image_channels: Number of channels for the input images
            init_image_size: The initial size of the images fed into the MRUs
            image_pool: Pooling function to halve size of images fed into MRUs
            **kwargs: Keyword arguments to pass into MRU layer
                norm: Normalization function to use after convolution (without num_features specified)
                mask_norm: True if norm function should also be applied to masks (mi and ni)
                activation: Activation function, f, to use
                mask_activation: Activation function to use on masks mi and ni
                sn: True if spectral normalization is used for convolution weights (mainly for discriminator)
        """
        super(Generator, self).__init__()
        self.label_embeds = nn.Embedding(num_classes, init_image_size ** 2)
        self.init_image_size = init_image_size
        self.encoder = Encoder(num_classes, 1, init_out_channels=init_out_channels, image_channels=image_channels,
                               init_image_size=init_image_size, image_pool=image_pool, **kwargs)
        self.decoder = Decoder(init_out_channels=init_out_channels, image_channels=image_channels, 
                               init_image_size=init_image_size, image_pool=image_pool, **kwargs)
        
    
    def forward(self, labels, images):
        """
        Implements forward pass of encoder
        
        Args:
            labels: Tensor of integer labels for each image of size (batch_size, 1)
            images: Tensor of images of size (batch_size, image_channels, init_image_size, init_image_size)
            
        Returns:
            out: Output from the decoder, with size (batch_size, 3, init_image_size, init_image_size)
            noise: The Gaussian noise concatenated at the bottleneck
        """
        ## Convert labels from (batch_size, 1) to size (batch_size, 1, init_image_size, init_image_size)
        embeds = self.label_embeds(labels)
        embeds = embeds.view(-1, 1, self.init_image_size, self.init_image_size)
        
        encoder_output = self.encoder((embeds, images))
        return self.decoder((encoder_output, images))
