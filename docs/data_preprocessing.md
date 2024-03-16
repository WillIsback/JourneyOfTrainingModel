This document explains the code in `data_preprocessing.py` which is used for preprocessing the data for a caption generation model.

## Importing Libraries

```Python
import nltk
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import os
import json
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
from Utils.gpu_utils import get_gpu_memory
import logging
```

The code begins by importing the necessary libraries. These include `nltk` for natural language processing, `torch` and `torchvision` for deep learning, `PIL` for image processing, and `logging` for logging GPU memory usage.

## Setting Up Logging

```Python
logging.basicConfig(filename='gpu_memory_usage.log', level=logging.INFO)
```

This line sets up logging to a file named `gpu_memory_usage.log`. The logging level is set to `INFO`, meaning that all messages of level `INFO` and above will be logged.

## Vocabulary Class

```Python
class Vocabulary(object):
    ...
```

This class is used to create a vocabulary, which is a mapping between words and numerical indices. This is necessary for converting the captions into a format that can be processed by the model.

## Building the Vocabulary

```Python
def build_vocab(caption_list, threshold=4):
    ...
```

This function builds the vocabulary by tokenizing the captions and counting the frequency of each word. Words that appear at least `threshold` times are added to the vocabulary.

## Image Preprocessing

```Python
def resize_image(image, size):
    ...
def preprocess_image(image_path):
    ...
```

These functions are used to preprocess the images. `resize_image` resizes an image to a specified size, and `preprocess_image` applies a series of transformations to an image, including resizing, cropping, flipping, and normalization.

## Caption Dataset Class

```Python
class CaptionDataset(Dataset):
    ...
```

This class is a subclass of `torch.utils.data.Dataset` and is used to load the images and captions. The `__getitem__` method returns the preprocessed image and the corresponding caption as a tensor of word indices.

## Collating Function

```Python
def collate_fn(data):
    ...
```

This function is used to batch the data. It sorts the data by caption length in descending order, stacks the images into a single tensor, and pads the captions so that they all have the same length.

## Loading the Data

```Python
with open("DataSet/annotations/captions_train2017.json", 'r') as f:
    ...
with open("DataSet/annotations/captions_val2017.json", 'r') as f:
    ...
```

These lines load the annotations files, which contain the captions for the images.

## Creating the Datasets and Data Loaders

```Python
vocab = build_vocab(caption_list_train)
...
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers, 
                          collate_fn=collate_fn)
...
```

These lines create the vocabulary, datasets, and data loaders. The data loaders are used to load the data in batches during training.

## Checking for GPU Availability

```Python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

This line checks if a GPU is available for training. If a GPU is available, it sets `device` to `'cuda'`, otherwise it sets `device` to `'cpu'`.

## Logging GPU Memory Usage

```Python
def Log_gpu_memory_usage(epoch):
    ...
```

This function logs the GPU memory usage at each epoch. This can be useful for monitoring memory usage during training.