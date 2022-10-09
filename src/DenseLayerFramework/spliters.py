"""
Spliters take the whole set we have for training and split the data somehow into 


"""
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random

class Spliter(ABC):

  @abstractmethod
  def __init__(self, X, Y,  *args, **kwargs):
    assert len(X) == len(Y)
    self.X = X
    self.Y = Y
  
  @abstractmethod
  def split(self, *args, **kwargs):
    pass


class SimpleSpliter(Spliter):

  def __init__(self, X, Y, test_size:float=0.25, *args, **kwargs):
     super(SimpleSpliter, self).__init__(X, Y)
     self.test_size = test_size

  def split(self, *args, **kwargs):
    return train_test_split(self.X, self.Y, test_size=self.test_size)


class NFromEveryClassSpliter(Spliter):

  def __init__(self, X, Y, n:int=1, *args, **kwargs):
     super(NFromEveryClassSpliter, self).__init__(X, Y)
     label, count = np.unique(Y, return_counts=True)
     assert all([c>n for c in count]), "Not all classes have more than n samples"
     self.n = n
     self.classes = np.unique(Y)

  def split(self, *args, **kwargs):
    val_ids = []
    for label in self.classes:
      label_ids = np.argwhere(self.Y == label).flatten()
      choosen_ids = random.choice(label_ids, size=self.n, replace=False)
      val_ids.extend(choosen_ids)
    train_ids = [i for i in range(len(self.Y)) if i not in val_ids]
    return (self.X[train_ids],
           self.X[val_ids],
           self.Y[train_ids],
           self.Y[val_ids])


class FractionFromEveryClassSpliter(Spliter):

  def __init__(self, X, Y, fraction:float=0.25, *args, **kwargs):
     super(FractionFromEveryClassSpliter, self).__init__(X, Y)
     labels, counts = np.unique(Y, return_counts=True)
     self.val_sizes = {label: int(np.ceil(count*fraction)) for label, count in zip(labels, counts)}
     self.classes = labels

  def split(self, *args, **kwargs):
    val_ids = []
    for label in self.classes:
      label_ids = np.argwhere(self.Y == label).flatten()
      val_size = self.val_sizes[label]
      choosen_ids = random.choice(label_ids, size=val_size, replace=False)
      val_ids.extend(choosen_ids)
    train_ids = [i for i in range(len(self.Y)) if i not in val_ids]
    return (self.X[train_ids],
           self.X[val_ids],
           self.Y[train_ids],
           self.Y[val_ids])

