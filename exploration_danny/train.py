"""
Training process of SketchyGAN
Mostly adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from generator import *
from discriminator import *
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from utils import * 
from loss import *


#PARAMS

# Learning rates for discriminator and generator based on the SketchyGAN paper
d_lr = 0.0002
g_lr = 0.0001

num_epochs = 5
batch_size = 8


# TODO: implement photo_sketch_dl 
photo_sketch_dl = None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = Generator(num_classes=125,
                 activation=nn.ReLU,
                 norm=nn.BatchNorm2d,
                 init_out_channels=64,
                 image_channels=3,
                 init_image_size=64
                 ).to(device)

netD = Discriminator(num_classes=125
                     activation=nn.LeakyReLU(negative_slope=0.1, inplace=True),
                     norm=nn.BatchNorm2d,
                     init_out_channels=64,
                     image_channels=3,
                     init_image_size=64
                     ).to(device)

dis_criterion = nn.BCEWithLogitsLoss()
aux_criterion = nn.CrossEntropyLoss()

# import supervision, perceptual, and divsersity loss
spd_loss = None

real_label = 1.0
fake_label = 0.0


optimizerD = optim.Adam(netD.parameters(), lr=d_lr)
optimizerG = optim.Adam(netG.parameters(), lr=g_lr)
# TODO: Maybe add schedulers for optimizers?

G_losses = []
D_losses = []
img_list = []
iters = 0

print("Begin training ...")
begin_time = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(photo_sketch_dl):
        """
        Loss function(criterion) returns sum of 
        GANLoss - auxiliary loss + supervision loss + perceptual loss + diversity loss
        """
        # Assuming data is a list of [photos, labels, sketches]
        input_photos, input_sketches, input_labels = data
        
        ## Update netD: maximize log(D(y)) + log(1-D(G(x,z)))
        # Train with all-real images
        netD.zero_grad()
        # Format batch
        dis_label = torch.full((batch_size, ), real_label, device=device)
        aux_label = torch.full((batch_size, ), input_labels, device=device)
        # Forward pass real batch thru D
        dis_output_real, aux_output_real = netD(input_photos)
        # Calculate loss on all-real batch
        dis_errD_real = dis_criterion(dis_output_real, dis_label) #pred_real_natural
        aux_errD_real = aux_criterion(aux_output_real, aux_label) #pred_real_class
        # Calculate gradients for D in the backward pass
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.mean().item()
        
        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        ## Train with all-fake batch
        # Generate fake image batch with G 
        # Generator outputs generated image and noise vector applied on the bottleneck
        fake, z = netG(input_labels, input_sketches)
        dis_label.data.fill_(fake_label)
        # Classify all fake batch with D
        dis_output_fake, aux_output_fake = netD(fake)
        # Calculate D's loss on the all-fake batch
        dis_errD_fake = dis_criterion(dis_output_fake, dis_label) #pred_fake_natural
        aux_errD_fake = aux_criterion(aux_output_fake, aux_label) #pred_fake_class
        # Calculate the gradients for this batch
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.mean().item()
        # Calculate DRAGAN Loss
        grad_penalty = gradient_penalty(netD, input_photos, fake)
        grad_penalty.backward()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake + grad_penalty
        # Update D
        optimizerD.step()

        ## Update G Network: maximize log(D(G(x,z)))
        netG.zero_grad()
        dis_label.data.fill_(real_label) #fake labels are real label for generator
        # Since we just updated D, perform another forward pass of all-fake batch thru D
        dis_output, aux_output = netD(fake)
        # Calculate G's loss based on this output
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        # Calculate gradients for G
        # Supervised loss
        supervised_loss_G = supervised_loss(fake, input_photos)
        # Perceptual loss
        perceptual_loss_G = perceptual_loss(fake, input_photos)
        # Diversity Loss; Create another fake image
        fake_alt , z_alt = netG(input_labels, input_sketches)
        diversity_loss_G = diversity_loss(fake, fake_alt, z, z_alt)
        errG = dis_errG - aux_errG + supervised_loss_G + perceptual_loss_G + diversity_loss_G
        errG.backward()
        D_G_z2 = dis_output.mean().item()        
        # Update G
        optimizerG.step()

        # Print trainig stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(photo_sketch_dl),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                )
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Save a sketch and corresponding photo
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(photo_sketch_dl) - 1)):
            img_list.append((fake[0], input_photos[0]))

                
print("Training ended.")
print("Time took : ", time.time()-begin_time)


# Plot losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot sketches and photos pairs saved during the training
