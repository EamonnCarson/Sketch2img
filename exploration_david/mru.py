import torch
import torch.nn as nn

class MRU(nn.Module):
    """
    Masked Residual Unit, which takes in feature maps x_i and an image I, and outputs new feature maps y_i
    """
    def _conv3x3(self, in_channels, out_channels, norm, activation):
        """
        Implements 3x3 convolution, normalization, and activation in MRU unit.
        
        Args:
            in_channels: Number of input channels (e.g. 3 for a RGB image)
            out_channels: Number of output channels
            norm: Normalization function to use (without num_features specified). Can be None for no norm.
            activation: Activation function to use after convolution
        """
        # Padding of 1 ensures that input size = output size
        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight)
        if self.sn:
            conv = torch.nn.utils.spectral_norm(conv)
        ml = nn.ModuleList([conv])
        
        if norm:
            ml.append(norm(out_channels))  # Give it correct num_features, then feed in input
        ml.append(activation)
        return ml
    
    def __init__(self, in_channels, out_channels, image_channels, norm=nn.BatchNorm2d, mask_norm=False,
                 activation=nn.ReLU(), mask_activation=nn.Sigmoid(), sn=False):
        """
        Args:
            in_channels: Number of input channels (e.g. 3 for a RGB image)
            out_channels: Number of output channels
            image_channels: Number of channels for the input image
            norm: Normalization function to use after convolution (without num_features specified)
            mask_norm: True if norm function should also be applied to masks (mi and ni)
            activation: Activation function, f, to use
            sn: True if spectral normalization is used for convolution weights (mainly for discriminator)
        """
        super(MRU, self).__init__()
        mask_norm = norm if mask_norm else None
        self.sn = sn
        self.conv_mi = self._conv3x3(in_channels + image_channels, in_channels, mask_norm, mask_activation)
        self.conv_ni = self._conv3x3(in_channels + image_channels, out_channels, mask_norm, mask_activation)
        self.conv_zi = self._conv3x3(in_channels + image_channels, out_channels, norm, activation)
        self.conv_xi = self._conv3x3(in_channels, out_channels, norm, activation)

    def forward(self, xi, image):
        """
        Args:
            xi: Input feature map of size (batch_size, in_channels, height, width)
            image: Input images of size (batch_size, sketch_channels, height, width)
            
        Returns:
            yi: Output feature map of size (batch_size, out_channels, height, width)
        """
        xi_image_concat = torch.cat((xi, image), dim=1)  # Concat on channel dimension
    
        mi = self.conv_mi(xi_image_concat)
        mask_concat = torch.cat((mi * xi, image), dim=1)
        zi = self.conv_zi(mask_concat)
        
        ni = self.conv_ni(xi_image_concat)
        xi_new = self.conv_xi(xi)
        
        yi = (1 - ni) * xi_new + ni * zi
        return yi