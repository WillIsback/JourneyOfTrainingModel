# data_preprocessing.py
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
#nltk.download('punkt')
# Vocabulary class for mapping between words and numerical indices
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# Function to build vocabulary
def build_vocab(caption_list, threshold=4):
    counter = nltk.Counter()
    for i, caption in enumerate(caption_list):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

# Function to resize an image
def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

# Function to preprocess an image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Dataset class for captions
class CaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)['annotations']
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        img_id = self.captions[idx]['image_id']
        image = Image.open(os.path.join(self.image_dir, f'{img_id:012d}.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.LongTensor(caption)
        return image, target

def collate_fn(data):
    # Sort data by caption length in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions = zip(*data)

    # Stack images into a single tensor
    images = torch.stack(images, 0)

    # Pad captions and convert to LongTensor
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

# Load the annotations file
with open("DataSet/annotations/captions_train2017.json", 'r') as f:
    annotations_train = json.load(f)
    # Load the annotations file
with open("DataSet/annotations/captions_val2017.json", 'r') as f:
    annotations_val = json.load(f)

# Extract the captions
caption_list_train = [anno['caption'] for anno in annotations_train['annotations']]
caption_list_val = [anno['caption'] for anno in annotations_val['annotations']]

# Define your variables here
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])
batch_size = 32  # Batch size
num_workers = os.cpu_count()  # Number of workers for data loading
collate_fn = collate_fn  # Function to batch data

# Create the vocabulary
vocab = build_vocab(caption_list_train)

# Create the datasets
image_dir_train = "DataSet/train2017"
image_dir_val = "DataSet/val2017"
captions_file_train = "DataSet/annotations/captions_train2017.json"
captions_file_val = "DataSet/annotations/captions_val2017.json"
dataset_train = CaptionDataset(image_dir_train, captions_file_train, vocab, transform)
dataset_val = CaptionDataset(image_dir_val, captions_file_val, vocab, transform)

# Split the training dataset into training and test sets
train_size = int(0.8 * len(dataset_train))
test_size = len(dataset_train) - train_size
train_dataset, test_dataset = random_split(dataset_train, [train_size, test_size])

# Assuming `train_dataset` is your original training dataset
# train_subset = Subset(train_dataset, range(100))  # Use the first 100 data points
# train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

# Create the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers, 
                          collate_fn=collate_fn)

val_loader = DataLoader(dataset=dataset_val, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers, 
                        collate_fn=collate_fn)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=True, 
                         num_workers=num_workers, 
                         collate_fn=collate_fn)