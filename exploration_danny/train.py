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


#PARAMS

# Learning rates for discriminator and generator based on the SketchyGAN paper
d_lr = 0.0002
g_lr = 0.0001

num_epochs = 5
batch_size = 8


# TODO: implement photo_sketch_dl 
photo_sketch_dl = None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = Encoder(num_classes=125,
               activation=nn.ReLU,
               norm=nn.BatchNorm2d,
               init_out_channels=64,
               image_channels=3,
               init_image_size=64
               ).to(device)

netD = Discriminator(activation=nn.LeakyReLU(negative_slope=0.1, inplace=True),
                     norm=nn.BatchNorm2d,
                     init_out_channels=64,
                     image_channels=3,
                     init_image_size=64
                     ).to(device)

dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# import supervision, perceptual, and divsersity loss
spd_loss = None

real_label = 1
fake_label = 0


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
        input_photos, input_labels, input_sketches = data
        
        ## Update netD: maximize log(D(y)) + log(1-D(G(x,z)))
        # Train with all-real images
        netD.zero_grad()
        # Format batch
        dis_label = torch.full((batch_size, ), real_label, device=device)
        aux_label = torch.full((batch_size, ), input_labels, device=device)
        # Forward pass real batch thru D
        # TODO: figure out first argument to netD
        dis_output, aux_output = netD(..., input_photos)
        # Calculate loss on all-real batch
        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        # Calculate gradients for D in the backward pass
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.mean().item()
        
        # TODO: implement compute_acc
        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        ## Train with all-fake batch
        # Generate fake image batch with G 
        # Generator outputs generated image and noise vector applied on the bottleneck
        fake, z = netG(input_labels, input_sketches)
        dis_label.data.fill_(fake_label)
        # Classify all fake batch with D
        #TODO: fill first arg netD
        dis_output, aux_output = netD(..., fake.detach())
        # Calculate D's loss on the all-fake batch
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        # Calculate the gradients for this batch
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = disoutput.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ## Update G Network: maximize log(D(G(x,z)))
        netG.zero_grad()
        dis_label.data.fill_(real_label) #fake labels are real label for generator
        # Since we just updated D, perform another forward pass of all-fake batch thru D
        # TODO: fill first arg
        dis_output, aux_output = netD(..., fake)
        # Calculate G's loss based on this output
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        # Calculate gradients for G
        additional_loss = spd_loss(fake, input_photos, z)
        errG = dis_errG - aux_errG + additional_loss
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
