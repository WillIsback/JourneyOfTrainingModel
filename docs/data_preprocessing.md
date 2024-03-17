This document explains the code in `data_preprocessing.py` which is used for preprocessing the data for a caption generation model.


The code begins by importing the necessary libraries. These include `nltk` for natural language processing, `torch` and `torchvision` for deep learning, `PIL` for image processing, and `logging` for logging GPU memory usage.

## Import [COCO](https://cocodataset.org/#download) 2017 DataSet (corpus) setup
To set up the dataset for this project, you can use the provided `download_data.sh` script. This script will automatically download and unzip the necessary datasets.
<br><br>
To run the script, open a terminal and navigate to `downloader` directory containing the script. Then, run the following command:
<details>
<summary>Click to expand!</summary>

```bash
#!/bin/bash

# Create the DataSet directory if it doesn't exist
mkdir -p ../DataSet
cd ../DataSet

# Download the datasets
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the datasets
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip

# Clean up the zip files
rm annotations_trainval2017.zip
rm train2017.zip
rm val2017.zip
rm test2017.zip

cd ..
```

</details>

```bash
./download_data.sh
```


## Vocabulary Class

```python
# This is a Vocabulary class used for creating a mapping between words and numerical indices.
# This is a common technique used in Natural Language Processing (NLP) to convert text data into numerical form that can be understood by machine learning algorithms.

class Vocabulary(object):
```
<details>
<summary>Click to expand!</summary>

```Python
    def __init__(self):
        # word2idx is a dictionary that will hold the mapping from words to unique indices.
        self.word2idx = {}
        # idx2word is a dictionary that will hold the mapping from indices back to the corresponding words.
        # This is useful for converting the output of algorithms back into a human-readable format.
        self.idx2word = {}
        # idx is a counter used to assign a unique index to each new word that is added to the vocabulary.
        self.idx = 0

    def add_word(self, word):
        # This method adds a word to the vocabulary.
        # If the word is not already in the vocabulary, it assigns it a unique index and adds it to the word2idx and idx2word dictionaries.
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        # This method allows the Vocabulary object to be callable.
        # When called with a word as an argument, it returns the index of the word.
        # If the word is not in the vocabulary, it returns the index of the '<unk>' token, which stands for 'unknown'.
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        # This method allows the use of the len() function on a Vocabulary object.
        # When called, it returns the number of words in the vocabulary.
        return len(self.word2idx)
```

</details>

<br>This class is used to create a vocabulary, which is a mapping between words and numerical indices. This is necessary for converting the captions into a format that can be processed by the model.

## Building the Vocabulary

```python
# This function is used to build a vocabulary from a list of captions.
# The vocabulary is a mapping of words to unique indices that is used to convert text data into a numerical form that can be understood by machine learning algorithms.

def build_vocab(caption_list, threshold=4):
    ...
```
<details>
<summary>Click to expand!</summary>

```Python
    # nltk.Counter is a dictionary subclass for counting hashable objects.
    # Here it is used to count the occurrence of each word in the captions.
    counter = nltk.Counter()
    for i, caption in enumerate(caption_list):
        # The captions are tokenized into words using nltk's word_tokenize function.
        # The words are also converted to lowercase to ensure that the vocabulary is case-insensitive.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        # The counter is updated with the new words.
        counter.update(tokens)

    # A list of words is created that includes only the words that occur at least 'threshold' times in the captions.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # A Vocabulary object is created.
    vocab = Vocabulary()
    # Special tokens that are used in sequence-to-sequence models are added to the vocabulary.
    # '<pad>' is used for padding shorter sequences to a common length.
    # '<start>' and '<end>' are used to indicate the beginning and end of a sequence.
    # '<unk>' is used to represent unknown words that are not in the vocabulary.
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # The words from the list are added to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    # The completed vocabulary is returned.
    return vocab
```

</details>

<br>
This function builds the vocabulary by tokenizing the captions and counting the frequency of each word. Words that appear at least `threshold` times are added to the vocabulary.

## Loading and Preprocessing the Data

```python
def load_and_preprocess_data():
    ...
```

This function is used to load and preprocess the data. It first defines a series of transformations to apply to the images, including resizing, cropping, flipping, and normalization. Then, it loads the annotations files, which contain the captions for the images. It extracts the captions and uses them to build the vocabulary.

Next, it checks if the tar files for the preprocessed data already exist. If they don't, it creates them. It opens each image, applies the transformations, and saves the image and its corresponding caption to the tar file. The captions are tokenized and converted to numerical indices using the vocabulary.

Finally, it creates a `WebDataset` for the training and validation data. The `WebDataset` class is a PyTorch-compatible dataset for loading data from tar files. It returns the training dataset, validation dataset, and the vocabulary.

## NLTK Download

```python
nltk.download('punkt')
```

This line downloads the Punkt tokenizer models. This is necessary for tokenizing the captions in the `build_vocab` function.

## Import Statements

```python
import nltk
import torch
from torchvision import transforms
import os
import json
from Utils.Utils import get_absolute_path
import logging
from PIL import Image
from webdataset import WebDataset
from tqdm import tqdm
```

These lines import the necessary modules for the script. `nltk` is used for tokenizing the captions, `torch` and `torchvision` are used for creating the datasets and applying transformations to the images, `os` and `json` are used for file handling, `get_absolute_path` is a utility function for getting the absolute path of a file, `logging` is used for logging, `Image` is used for opening and converting the images, `WebDataset` is used for creating the datasets, and `tqdm` is used for displaying progress bars.
