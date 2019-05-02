import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MRU(nn.Module):
    def conv3x3(in_channels, out_channels):
        """
        Implements 3x3 convolution layer in MRU unit.
        """
        return nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1,  # Padding of 1 ensures that input size = output size
    
    def __init__(self, in_channels, out_channels, activation, sketch_channels=3):
        """
        Initializes a MRU unit for SketchyGAN.
        
        Args:
            in_channels: Number of input channels (e.g. 3 for a RGB image)
            out_channels: Number of output channels
            activation: Activation function, f, to use
            sketch_channels: Number of channels for the input sketch
        """
        super(MRU, self).__init__()
        self.conv_mi = conv3x3(in_channels + sketch_channels, in_channels)
        self.conv_ni = conv3x3(in_channels + sketch_channels, out_channels)
        self.conv_zi = conv3x3(in_channels + sketch_channels, out_channels)
        self.conv_xi = conv3x3(in_channels, out_channels)
        self.f = activation  # activation needs to be in nn.Module

    def forward(self, xi, image):
        """
        Implement forward pass of MRU.
        
        Args:
            xi: Input feature map of size (batch_size, in_channels, height, width)
            image: Input sketch of size (batch_size, sketch_channels, height, width)
            
        Returns:
            yi: Output feature map of size (batch_size, out_channels, height, width)
        """
        xi_image_concat = torch.cat((xi, image), dim=1)  # Concat on channel dimension
    
        mi = nn.Sigmoid(self.conv_mi(xi_image_concat))
        mask_concat = torch.cat((mi * xi, image), dim=1)
        zi = self.f(self.conv_zi(mask_concat))
        
        ni = nn.Sigmoid(self.conv_ni(xi_image_concat))
        xi_new = self.conv_xi(xi)
        
        yi = (1 - ni) * zi + ni * xi_new
        return yi

class sketchGAN_G(nn.Moduel):
    def __init__(self, mru, layer_dims, sampling_dims):
        # layer_dims is a list of out_channels for each layer
        # sampling_dims is a list of output shape of each layer (y)
        super(sketchGAN_G, self).__init__()
        self.in_channels = None # TODO :: set to initial channel
        # encoding layers
        self.enc_layer1 = self.make_layer(mru, layer_dims[0], interpoloate=sampling_dims[0])
        self.enc_layer2 = self.make_layer(mru, layer_dims[1], interpoloate=sampling_dims[1])
        self.enc_layer3 = self.make_layer(mru, layer_dims[2], interpoloate=sampling_dims[2])
        self.enc_layer4 = self.make_layer(mru, layer_dims[3], interpoloate=sampling_dims[3])
        # decoding layers
        self.dec_layer1 = self.make_layer(mru, layer_dims[4], interpoloate=sampling_dims[4])
        self.dec_layer2 = self.make_layer(mru, layer_dims[5], interpoloate=sampling_dims[5])
        self.dec_layer3 = self.make_layer(mru, layer_dims[6], interpoloate=sampling_dims[6])
        self.dec_layer4 = self.make_layer(mru, layer_dims[7], interpoloate=sampling_dims[7])
    
    def make_layer(self, mru, out_channels, activation=nn.ReLU, stride=1, interpoloate=None):
        layer = mru(self.in_channels, out_channels, activation=activation, stride=stride, interpoloate=interpoloate)
        self.in_channels = out_channels
        return nn.Sequential(*layer)

    def forward(self, x):
        out1 = self.enc_layer1(x)
        out2 = self.enc_layer2(out1)
        out3 = self.enc_layer3(out2)
        out4 = self.enc_layer4(out3)
        out5 = self.dec_layer1(out4)
        # out5 = torch.cat((out5, out4), axis=1) 
        # Not sure this is needed. This skip-connection is not shown in figure diagram
        out6 = self.dec_layer2(out5)
        out6 = torch.cat((out6, out3), axis=1)
        out7 = self.dec_layer3(out6)
        out7 = torch.cat((out7, out2), axis=1)
        out = self.dec_layer4(out7)
        out = torch.cat((out, out1), axis=1)
        return out

# model = sketchGAN_G(MRU, layer_dims, sampling_dims).to(device)