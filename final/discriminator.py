import torch
import torch.nn as nn
from generator import Encoder

class Discriminator(nn.Module):
    """
    Discriminator for SketchyGAN, which is mostly based on the encoder block
    """
    def __init__(self, num_classes, init_in_channels, init_out_channels=64, init_image_size=64, 
                 image_pool=nn.AvgPool2d(2, stride=2), **kwargs):
        super(Discriminator, self).__init__()
        # Use image_channels default value of 3
        self.encoder = Encoder(num_classes, init_in_channels, init_out_channels=init_out_channels, 
                               init_image_size=init_image_size, image_pool=image_pool, **kwargs)
        self.linear_size = int(init_out_channels * 8 * (init_image_size / 8) ** 2)
        self.fc_dis = nn.Linear(self.linear_size, 1)
        self.fc_aux = nn.Linear(self.linear_size, num_classes)
        
    def forward(self, image):
        out = self.encoder.forward(image, image)
        out = out.view(-1, linear_size)
        dis_out = self.fc_dis(out)
        aux_out = self.fc_aux(out)
        return dis_out, aux_out