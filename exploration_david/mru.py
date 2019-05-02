import torch
import torch.nn as nn

class MRU(nn.Module):
    """
    Masked Residual Unit, which takes in feature maps x_i and an image I, and outputs new feature maps y_i
    """
    @staticmethod
    def _conv3x3(in_channels, out_channels, sn):
        """
        Implements 3x3 convolution layer in MRU unit.
        
        Args:
            in_channels: Number of input channels (e.g. 3 for a RGB image)
            out_channels: Number of output channels
            sn: Set True to use spectral normalization (mainly for discriminator)
        """
        # Padding of 1 ensures that input size = output size
        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight)
        if sn:
            conv = torch.nn.utils.spectral_norm(conv)
        return conv
    
    def __init__(self, in_channels, out_channels, image_channels, activation, sn=False):
        """
        Args:
            in_channels: Number of input channels (e.g. 3 for a RGB image)
            out_channels: Number of output channels
            image_channels: Number of channels for the input image
            activation: Activation function, f, to use
            sn: Set True to use spectral normalization (mainly for discriminator)
        """
        super(MRU, self).__init__()
        self.conv_mi = self._conv3x3(in_channels + image_channels, in_channels, sn)
        self.conv_ni = self._conv3x3(in_channels + image_channels, out_channels, sn)
        self.conv_zi = self._conv3x3(in_channels + image_channels, out_channels, sn)
        self.conv_xi = self._conv3x3(in_channels, out_channels, sn)
        self.f = activation  # activation needs to be in nn.Module

    def forward(self, xi, image):
        """
        Args:
            xi: Input feature map of size (batch_size, in_channels, height, width)
            image: Input images of size (batch_size, sketch_channels, height, width)
            
        Returns:
            yi: Output feature map of size (batch_size, out_channels, height, width)
        """
        xi_image_concat = torch.cat((xi, image), dim=1)  # Concat on channel dimension
    
        mi = nn.Sigmoid(self.conv_mi(xi_image_concat))
        mask_concat = torch.cat((mi * xi, image), dim=1)
        zi = self.f(self.conv_zi(mask_concat))
        
        ni = nn.Sigmoid(self.conv_ni(xi_image_concat))
        xi_new = self.conv_xi(xi)
        
        yi = (1 - ni) * xi_new + ni * zi
        return yi