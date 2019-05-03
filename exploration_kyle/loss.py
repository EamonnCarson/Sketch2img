import torch
import torchvision
import numpy as np
import scipy 

def supervised_loss(image_generated, image_true):
    loss = torch.nn.L1Loss()
    return loss(image_generated, image_true)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
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
    return -torch.mean(supervised_loss(image_generated_a, image_generated_b) / torch.norm(noise_a - noise_b, axis=1).view(-1, 1, 1, 1))

def generator_loss(image_true, image_generated_a, noise_a, labels_a, image_generated_b, noise_b, labels_b):
    # this might need more arguments depending on how the gan_loss and ac_loss are implemented 
    #TODO: populate gan_loss and ac_loss
    return gan_loss - ac_loss + supervised_loss(image_generated_a, image_true) + perceptual_loss(image_generated_a, image_true) + diversity_loss(image_generated_a, image_generated_b, noise_a, noise_b)