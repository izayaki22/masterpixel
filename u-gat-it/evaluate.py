import torch
from networks import ResnetGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
from tqdm import tqdm
import matplotlib.pyplot as plt

class FIDDataset(torch.utils.data.Dataset):
    def __init__(self, files, device, translator, transform=None, original_domain=True):
        self.files = files
        self.device = device
        self.transforms = transform
        self.translator = translator
        self.original_domain = original_domain

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        if self.original_domain:
            return img.to(self.device).detach()
        else:
            translated_img, _, _ = self.translator(img.unsqueeze(0).to(self.device))
            return translated_img[0].detach()

def get_translator(ckpt_path):
    # UGATIT translation
    checkpoint = torch.load(ckpt_path)
    G_BA = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=False)
    G_BA.load_state_dict(checkpoint['genB2A'])
    G_BA.eval()
    return G_BA


def compute_statistics_of_path(path, model, translator, original_domain, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, translator, original_domain, batch_size,
                                               dims, device, num_workers)

    return m, s

def calculate_activation_statistics(files, model, translator, original_domain, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- translator  : Instance of translation model
    -- original_domain : if it is getting activation map of original domain 
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, translator, original_domain, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_activations(files, model, translator, original_domain, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- translator  : Instance of translation model
    -- original_domain : if it is getting activation map of original domain 
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # train 시와 같은 transform
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = FIDDataset(files, device=device, translator=translator, original_domain=original_domain, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        #breakpoint()

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_fid_given_paths(paths, batch_size, ckpt_path, device, dims, num_workers=1):
    """
    Calculates the FID of two paths

    paths = (testA_path, testB_path)
    ckpt_path = checkpoint path
    """
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    translator = get_translator(ckpt_path).to(device)
    
    with torch.no_grad():
        m1, s1 = compute_statistics_of_path(paths[0], model, translator, original_domain=True, batch_size=batch_size, # testA
                                            dims=dims, device=device, num_workers=num_workers)
        m2, s2 = compute_statistics_of_path(paths[1], model, translator, original_domain=False, batch_size=batch_size, # testB
                                            dims=dims, device=device, num_workers=num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


if __name__ == '__main__':
    sys.path.append('/root/MAIN')

    root = "/root/UGATIT-pytorch/result/pokemon-sprites"
    model_names = [name for name in os.listdir(os.path.join(root, 'saved_models'))]
    saved_steps = [int(name.split('_')[-1].split('.')[0]) for name in model_names]
    saved_steps = [step for step in saved_steps if step%1000 == 0]
    saved_steps.sort()

    target_domain_path = '/root/MAIN/data/pokemon-sprites/testA'
    source_domain_path = '/root/MAIN/data/pokemon-sprites/testB'
    paths = [target_domain_path, source_domain_path]
    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

    # setting
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 50
    dims = 2048 
    num_workers=1

    print("setting completed")
    print("steps:", saved_steps)
    print("device:", device)
    print("batch size:", batch_size)
    print("dimension:", dims)
    print("num_workers:", num_workers)

    fid_scores = []
    for step in saved_steps:
        model_name = f'pokemon-sprites_params_{step:07d}.pt'
        print(f"evaluating checkpoint {model_name}")
        fid_value = calculate_fid_given_paths(paths=paths,
                                          batch_size=batch_size,
                                          device=device,
                                          ckpt_path=os.path.join(root, 'saved_models', model_name),
                                          dims=dims,
                                          num_workers=num_workers)
        print("FID:", fid_value)
        fid_scores.append(fid_value)
    
    minidx = fid_scores.index(min(fid_scores))
    print("minimum epoch:", saved_epochs[minidx])
    print("minimum fid:", fid_scores[minidx])

    result_csv = pd.DataFrame({
        'step': saved_steps,
        'fid': fid_scores
    }).to_csv(os.path.join(root, 'result.csv'), index=False)
    
    plt.plot(saved_epochs, fid_scores, marker='o', linestyle='-', color='b', label='FID')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.xticks([saved_epochs[0], saved_epochs[-1]])
    plt.legend()
    plt.savefig(os.path.join(root, 'FIDscore.png'))