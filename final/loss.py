import torch
import torchvision
import numpy as np
import scipy
import logging
from torch.autograd import Variable, grad

def discriminator_loss(discriminator, real_image_sample, fake_image_sample, class_labels, lambda_dragan=10, k_dragan=1):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Generate basic predictions
    pred_real = discriminator.forward(real_image_sample) # predictions on real images
    pred_fake = discriminator.forward(fake_image_sample) # predictions on fake images
    pred_real_natural = pred_real[0] # prediction of real or fake
    pred_fake_natural = pred_fake[0]
    pred_real_class = pred_real[1] # prediction of which class
    pred_fake_class = pred_fake[1]

    # losses
    gan_loss_d_ = gan_loss_d(pred_real_natural, pred_fake_natural)
    gradient_penalty_ = gradient_penalty(discriminator, real_image_sample, fake_image_sample)
    ac_loss_d_ = ac_loss_d(pred_real_class, class_labels)
    
    logging.basicConfig(filename='discriminator.log', level=logging.DEBUG)
    logging.info("%f,%f,%f \n", gan_loss_d_, gradient_penalty_, ac_loss_d_)

    return gan_loss_d_ + gradient_penalty_ + ac_loss_d_

def generator_loss(discriminator, generator, real_image_sample, class_labels):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Generate fakes
    fake_image_sample, noise_a = generator.forward(real_image_sample)
    fake_image_sample_alt, noise_b = generator.forward(real_image_sample)

    # Get Discriminator Predictions
    pred_fake = discriminator.forward(fake_image_sample) # see discriminator
    pred_fake_natural = pred_fake[0]
    pred_fake_class = pred_fake[1]

    # losses
    gan_loss_g_ = gan_loss_g(pred_fake_natural)
    ac_loss_g_ = ac_loss_g(pred_fake_class, class_labels)
    supervised_loss_ = supervised_loss(fake_image_sample, real_image_sample)
    perceptual_loss_ = perceptual_loss(fake_image_sample, real_image_sample)
    diversity_loss_  = diversity_loss(fake_image_sample, fake_image_sample_alt, noise_a, noise_b)

    logging.basicConfig(filename='generator.log', level=logging.DEBUG)
    logging.info("%f,%f,%f,%f,%f \n", gan_loss_g_, ac_loss_g_, supervised_loss_, perceptual_loss_, diversity_loss_)

    return gan_loss_g_ + ac_loss_g_ + supervised_loss_ + perceptual_loss_ + diversity_loss_

# Discriminator component losses
def gan_loss_d(pred_real_natural, pred_fake_natural):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    # first evaluate loss on real images
    labels = torch.ones(pred_real_natural.size)
    loss_real = bce_loss(pred_real_natural, labels)
    # second evaluate loss on fake images
    labels = torch.zeros(pred_fake_natural.size)
    loss_fake = bce_loss(pred_fake_natural, labels)
    return loss_real + loss_fake

def gradient_penalty(discriminator, real_image_sample, fake_image_sample, lambda_=10, k=1):
    alpha = torch.rand(real_image_sample.size()[0], 1, 1, 1).expand(real_image_sample.size())
    interp = Variable(alpha * real_image_sample + (1 - alpha) * fake_image_sample, requires_grad=True)
    pred_hat, _ = discriminator.forward(interp)
    #pred_hat_natural = pred_hat[1]
    gradients = grad(outputs=pred_hat, inputs=interp, grad_outputs=torch.ones(pred_hat.size()),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = lambda_ * ((gradients.norm(2, dim=1) - k) ** 2).mean()
    return penalty

def ac_loss_d(pred_real_class, class_labels):
    ce_loss = torch.nn.CrossEntropyLoss()
    ac_loss = ce_loss(pred_real_class, class_labels)
    return ac_loss

# Generator component losses

def gan_loss_g(pred_fake_natural):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    labels = torch.ones(pred_fake_natural.size()) # ones because we want generator to fool discrim
    gan_loss = bce_loss(pred_fake_natural)
    return gan_loss

def ac_loss_g(pred_fake_class, class_labels):
    ce_loss = torch.nn.CrossEntropyLoss()
    ac_loss = ce_loss(pred_fake_class, class_labels)
    return ac_loss

def supervised_loss(image_generated, image_true):
    loss = torch.nn.L1Loss()
    return loss(image_generated, image_true)

def inception_score(imgs, cuda=False, batch_size=4, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    Credit: https://github.com/sbarratt/inception-score-pytorch
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.Tensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.Tensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(scipy.stats.entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def perceptual_loss(image_generated, image_true):
    return inception_score(image_generated)[0] - inception_score(image_true)[0], inception_score(image_generated)[1] - inception_score(image_true)[1]

def diversity_loss(image_generated_a, image_generated_b, noise_a, noise_b):
    return -torch.mean(supervised_loss(image_generated_a, image_generated_b) / torch.norm(noise_a - noise_b, dim=1).view(-1, 1, 1, 1))
