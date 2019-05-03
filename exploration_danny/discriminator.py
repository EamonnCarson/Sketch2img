import torch
import torch.nn as nn
import * from mru

class Discriminator(nn.Module):
    """
    Discriminator for SketchyGAN
    """
    @staticmethod
    def _mru_block(in_channels, out_channels, image_channels, activation, norm):
        return nn.Sequential(
            MRU(in_channels, out_channels, image_channels, activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),  # Halve height and width
            norm
        )
    
    def __init__(self, 
                 activation=nn.LeakyReLU(negative_slope=0.1, inplace=True),
                 norm=nn.BatchNorm2d,
                 init_out_channels=64,
                 image_channels=3,
                 init_image_size=64):
        """
        Initialize discriminator

        Args:
            activation: Activation function, f, to use
            norm: Normalization function to use inside encoder block
            init_out_channels: The number of channels the output of the first encoder block should output
            image_channels: Number of channels for the input images
            init_image_size: The initial size of the images fed into the MRUs
        """
        super(Discriminator, self).__init__()
        out1 = init_out_channels
        out2 = out1 * 2
        out4 = out2 * 2
        out8 = out4 * 2

        self.layer1 = self._mru_block(3, out1, image_channels, activation, norm(num_features=out1))
        self.layer2 = self._mru_block(out1, out2, image_channels, activation, norm(num_features=out2))
        self.layer3 = self._mru_block(out2, out4, image_channels, activation, norm(num_features=out4))
        self.layer4 = self._mru_block(out4, out8, image_channels, activation, norm(num_features=out8))
        self.downsampling1 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.downsampling2 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.downsampling3 = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.init_image_size = init_image_size
        self.last_layer = nn.Conv2d(out8, 1, kernel_size=4, stride=1, padding=0, bias=False) # Output 1 logit

    
    def forward(self, x, image):
        # Each batch must have all real images or all fake images input
        # x and image shape: 64x64
        out = self.layer1(x, image)
        image = self.downsampling1(image)
        # out and image shape: 32x32
        out = self.layer2(out, image)
        image = self.downsampling2(image)
        # out and image shape: 16x16
        out = self.layer3(out, image)
        image = self.downsampling3(image)
        # out and image shape: 8x8
        out = self.layer4(out, image)
        # out shape : 4x4
        out = self.last_layer(out)
        return out