import numpy as np
import os
import pandas as pd
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class InvertTransform:
    """A transform that takes a Tensor with values in [0, 1], and inverts those values."""
    def __call__(self, sample):
        return 1 - sample

class SketchyGanDataset(Dataset):
    """A custom Dataset class for SketchyGAN, combining images and sketches. """
    
    def generate_labels_id_map(self):
        """Returns a list of class labels"""
        d = {}
        for i, label in enumerate(sorted(os.listdir(self.photos_dir))):
            d[i] = label
            d[label] = i
        return d
    
    def __init__(self, photos_dir, sketches_dir, info_dir, photos_transform=None, sketches_transform=None,
                 remove_error=True, remove_ambiguous=False, remove_pose=False, remove_context=False):
        """
        Initialize the sketches dataset.
        
        Args:
            photos_dir (str): directory of photos
            sketches_dir (str): directory of sketches, divided by class
            info_dir (str): directory with additional information about the sketches
            remove_error (bool): set to True to remove sketches classified as erroneous
            remove_ambiguous (bool): set to True to remove sketches classified as ambiguous
            remove_pose (bool): set to True to remove sketches drawn from a wrong pose/perspective
            remove_context (bool): set to True to remove sketches with extraneous details
        """
        self.photos_dir = photos_dir
        self.sketches_dir = sketches_dir
        self.info_dir = info_dir
        self.labels_id_map = self.generate_labels_id_map()
        self.photos_transform = photos_transform
        self.sketches_transform = sketches_transform
        self.invalid = [line for line in open(os.path.join(info_dir, 'invalid-error.txt'), 'r')]
        self.stats = pd.read_csv(os.path.join(info_dir, 'stats.csv'))
        if remove_error:
            self.stats = self.stats.loc[self.stats['Error?'] == 0]
        if remove_ambiguous:
            self.stats = self.stats.loc[self.stats['Ambiguous?'] == 0]
        if remove_pose:
            self.stats = self.stats.loc[self.stats['WrongPose?'] == 0]
        if remove_context:
            self.stats = self.stats.loc[self.stats['Context?'] == 0]
    
    def __len__(self):
        return len(self.stats)
    
    def __getitem__(self, idx):
        row = self.stats.iloc[idx]
        class_folder = row['Category'].replace(' ', '_')
        sketch_file_str = f"{row['ImageNetID']}-{row['SketchID']}.png"
        sketch_path = os.path.join(self.sketches_dir, class_folder, sketch_file_str)
        with open(sketch_path, 'rb') as f:
            sketch = PIL.Image.open(f).convert('RGB')
        if self.sketches_transform:
            sketch = self.sketches_transform(sketch)
        
        photo_file_str = f"{row['ImageNetID']}.jpg"
        photo_path = os.path.join(self.photos_dir, class_folder, photo_file_str)
        with open(photo_path, 'rb') as f:
            photo = PIL.Image.open(f).convert('RGB')
        if self.photos_transform:
            photo = self.photos_transform(photo)
        
        return photo, sketch, self.labels_id_map[class_folder]
    
def load_sketchygan_dataset(batch_size):
    """
    Loads SketchyGAN dataset inside a DataLoader
    
    Args:
        batch_size: Batch size used by the DataLoader
    """
    npzfile = np.load("dataset_stats.npz")
    photos_mean = npzfile['photos_mean']
    photos_std = npzfile['photos_std']
    photos_scaling = npzfile['photos_scaling']
    sketches_mean = npzfile['sketches_mean']
    sketches_std = npzfile['sketches_std']
    sketches_scaling = npzfile['sketches_scaling']
    
    DATA_DIR = '../learning/datasets/sketchy/256x256'
    PHOTOS_AUG = 'tx_000100000000'
    PHOTOS_DIR = os.path.join(DATA_DIR, 'photo', PHOTOS_AUG)
    SKETCHES_AUG = 'tx_000000000010'
    SKETCHES_DIR = os.path.join(DATA_DIR, 'sketch', SKETCHES_AUG)
    INFO_DIR = '../learning/datasets/info-06-04/info'
    
    sketchygan_dataset = SketchyGanDataset(PHOTOS_DIR, SKETCHES_DIR, INFO_DIR, photos_transform=transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=photos_mean, std=np.array([photos_scaling, photos_scaling, photos_scaling]))
    ]), sketches_transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        InvertTransform(),
        transforms.Normalize(mean=sketches_mean, std=np.array([sketches_scaling, sketches_scaling, sketches_scaling]))
    ]))
    dl = DataLoader(sketchygan_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return sketchygan_dataset, dl