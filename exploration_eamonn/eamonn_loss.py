import torch
import torchvision
import numpy as np
import scipy

# code adapted from https://github.com/jfsantos/dragan-pytorch/blob/master/dragan.py

def gan_generator_loss(discriminator, generator, real_image_sample, sketch_sample):
    # first evaluate loss on real images (for discriminator)
    fake_image_sample = generator.forward(sketch_sample)
    pred_fake = discriminator.forward(fake_image_sample)
    labels = torch.ones(pred_fake.size) # ones because we want gen to fool discrim
    loss_fake = loss_func(pred_fake, labels)
    return loss_fake

def ac_generator_loss(discriminator, generator, real_image_sample, sketch_sample):
    ac_loss = ce_loss(pred_fake_class, class_labels)

def discriminator_loss(discriminator, real_image_sample, fake_image_sample, class_labels, lambda_=10, k_=1):
    bce_loss = torch.nn.BCELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Generate basic predictions
    pred_real = discriminator.forward(real_image_sample)
    pred_fake = discriminator.forward(fake_image_sample)
    pred_real_natural = pred_real[:, 0]
    pred_fake_natural = pred_fake[:, 0]
    pred_real_class = pred_real[:, 0:]
    pred_fake_class = pred_fake[:, 0:]

    # GAN Loss
    # first evaluate loss on real images (for discriminator)
    labels = torch.ones(pred_real_natural.size)
    loss_real = bce_loss(pred_real_natural, labels)
    # second evaluate loss on fake images (for discriminator)
    labels = torch.zeros(pred_fake_natural.size)
    loss_fake = bce_loss(pred_fake_natural, labels)
    # third do DRAGAN gradient penalty
    alpha = torch.rand(real_image_sample, 1)
    x_hat = Variable(alpha * real_image_sample + (1 - alpha) * (real_image_sample + 0.5 * real_image_sample.std() * torch.rand(real_image_sample.size())), requires_grad=True)
    pred_hat = discriminator.forward(x_hat)
    pred_hat_natural = pred_hat[:, 0]
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = lambda_ * ((gradients.norm(2, dim=1) - k) ** 2).mean()
    gan_loss = loss_real + loss_fake + gradient_penalty

    # AC Loss
    ac_loss = ce_loss(pred_real_class, class_labels)

    return ac_loss + gan_loss
