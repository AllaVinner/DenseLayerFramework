import torch 
from torch.utils.data import Dataset 
from numpy import random
import numpy as np
from numpy import random

# Data set classes
# Todo tomorrow <3 
def radialgamma_augmentation(x, s = 0.1, m = 1):
  k = np.power(m/s, 2)
  theta = np.power(s, 2)/m
  return random.gamma(k, theta)*x

class SinchDataset(Dataset):
  """Standard Dataset"""
  def __init__(self, x, y, *args, **kwargs):
    assert len(x) == len(y), "x and y need to have the same length"
    self.x = torch.from_numpy(x)
    self.y = torch.from_numpy(y)
  
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

class OversamplingDataset(Dataset):
  """Oversampling Dataset"""
  def __init__(self, x, y, *args, **kwargs):
    assert len(x) == len(y), "x and y need to have the same length"
    self.x: torch.Tensor = torch.tensor(x)
    self.y: torch.Tensor = torch.tensor(y)
    self.unique_labels, self.label_sizes = torch.unique(self.y, return_counts=True)
    self.labels_max_size = torch.max(self.label_sizes).item()
    self.num_labels = len(self.unique_labels)    
    self.num_samples = self.labels_max_size * self.num_labels
    cum_idx = torch.cumsum(self.label_sizes, 0)
    self.label_start_idx = torch.cat((torch.tensor([0]), cum_idx[0:-1])) 

  
  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    label_i = idx // self.labels_max_size
    sample_i = idx % self.labels_max_size
    if sample_i >= self.label_sizes[label_i]:
      sample_i = random.choice(self.label_sizes[label_i], 1)[0]
    
    sampled_idx = self.label_start_idx[label_i].item() + sample_i
    return self.x[sampled_idx], self.y[sampled_idx]


class OversamplingWithGammaAugmentationDataset(Dataset):
  """Oversampling Dataset"""
  def __init__(self, x, y, gamma_std = 0.1, *args, **kwargs):
    assert len(x) == len(y), "x and y need to have the same length"
    self.x: torch.Tensor = torch.tensor(x)
    self.y: torch.Tensor = torch.tensor(y)
    self.gamma_std = gamma_std
    self.unique_labels, self.label_sizes = torch.unique(self.y, return_counts=True)
    self.labels_max_size = torch.max(self.label_sizes).item()
    self.num_labels = len(self.unique_labels)    
    self.num_samples = self.labels_max_size * self.num_labels
    cum_idx = torch.cumsum(self.label_sizes, 0)
    self.label_start_idx = torch.cat((torch.tensor([0]), cum_idx[0:-1])) 

  
  def __len__(self):
    return self.num_samples
  
  def __getitem__(self, idx):
    label_i = idx // self.labels_max_size
    sample_i = idx % self.labels_max_size
    if sample_i >= self.label_sizes[label_i]:
      sample_i = random.choice(self.label_sizes[label_i], 1)[0]
    
    sampled_idx = self.label_start_idx[label_i].item() + sample_i
    x_aug = radialgamma_augmentation(self.x[sampled_idx], s = self.gamma_std, m = 1)
    return x_aug, self.y[sampled_idx]



