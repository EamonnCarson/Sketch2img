import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        pading=padding,
        bias=bias)

class MRU(nn.Module):
    def __init__(self, in_channels, out_channels, img_channels, activation, stride=1, downsample=False, upsample=False):
        super(MRU, self).__init__()
        self.conv1 = conv3x3(in_channels + img_channels, in_channels)
        self.sigmoid1 = nn.sigmoid(inplace=True)
        self.mul1 = torch.mul()
        self.activation = activation # activation needs to be in nn.module
        self.conv2 = conv3x3(in_channels + img_channels, out_channels)
        self.conv3 = conv3x3(in_channels + img_channels, out_channels)
        self.conv4 = conv3x3(in_channels, out_channels)
        self.sigmoid2 = nn.sigmoid(inplace=True)
        self.mul2 = torch.mul()
        self.mul3 = torch.mul()
        # Only one of upsample or downsample has to be True
        if upsample:
            self.resize = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        elif downsample:
            self.resize = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.resize = nn.Identity()

    def forward(self, x, image):
        out = torch.cat((x, image), axis=1) # axis=channel_index; 1 if (N,C,H,W)
        out = self.conv1(out)
        out = self.sigmoid1(out) # m
        out = self.mul1(out, x)
        out = torch.cat((out, image), axis=1)
        out = self.conv2(out)
        out = self.activation(out) # z
        n = torch.cat((x, image), axis=1)
        n = self.conv3(n) 
        n = self.sigmoid2(n)
        out_1 = self.mul2((torch.ones_like(n) - n), out)
        out_2 = self.mul3(n, x)
        out = out_1 + out_2
        out = self.resize(out)
        return out

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
        out6 = torch.cat((out6, out3), axis=1) # skip-connection
        out7 = self.dec_layer3(out6)
        out7 = torch.cat((out7, out2), axis=1) # skip-connection
        out = self.dec_layer4(out7)
        out = torch.cat((out, out1), axis=1) # skip-connection
        return out

# model = sketchGAN_G(MRU, layer_dims, sampling_dims).to(device)