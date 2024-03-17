# data_preprocessing.py
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

nltk.download('punkt') 




# This is a Vocabulary class used for creating a mapping between words and numerical indices.
# This is a common technique used in Natural Language Processing (NLP) to convert text data into numerical form that can be understood by machine learning algorithms.

class Vocabulary(object):
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

# This function is used to build a vocabulary from a list of captions.
# The vocabulary is a mapping of words to unique indices that is used to convert text data into a numerical form that can be understood by machine learning algorithms.

def build_vocab(caption_list, threshold=4):
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

    
    
# Function to load and preprocess the data----------------------------------------------------------    
def load_and_preprocess_data():
    # Define your variables here
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])

    # Load the training annotations file
    with open(get_absolute_path("DataSet/annotations/captions_train2017.json"), 'r') as f:
        annotations_train = json.load(f)
        
    # Load the validation annotations file
    with open(get_absolute_path("DataSet/annotations/captions_val2017.json"), 'r') as f:
        annotations_val = json.load(f)

    # Extract the captions
    caption_list_train = [anno['caption'] for anno in annotations_train['annotations']]
    caption_list_val = [anno['caption'] for anno in annotations_val['annotations']]

    # Create the vocabulary
    vocab = build_vocab(caption_list_train)

    # Create the training datasets
    image_dir_train = get_absolute_path("DataSet/train2017")

    # Create the validation datasets
    image_dir_val = get_absolute_path("DataSet/val2017")


    # Check if the tar files already exist
    if not os.path.exists("training.tar") or not os.path.exists("validation.tar"):
        # Create a tar file to store the preprocessed training data
        with open("training.tar", "wb") as outfile:
            for idx in tqdm(range(len(caption_list_train)), desc="Processing training data",dynamic_ncols=False, ncols=100):
                caption = caption_list_train[idx]
                img_id = annotations_train['annotations'][idx]['image_id']
                image_path = os.path.join(image_dir_train, f'{img_id:012d}.jpg')
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                torch.save(image, outfile)

                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                target = torch.LongTensor(caption)
                torch.save(target, outfile)

        # Create a tar file to store the preprocessed validation data
        with open("validation.tar", "wb") as outfile:
            for idx in tqdm(range(len(caption_list_val)), desc="Processing validation data",dynamic_ncols=False, ncols=100):
                caption = caption_list_val[idx]
                img_id = annotations_val['annotations'][idx]['image_id']
                image_path = os.path.join(image_dir_val, f'{img_id:012d}.jpg')
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                torch.save(image, outfile)

                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                target = torch.LongTensor(caption)
                torch.save(target, outfile)

    # Create a WebDataset for the training data
    training_dataset = WebDataset("training.tar").decode("pil").to_tuple("jpg;png", "json")

    # Create a WebDataset for the validation data
    validation_dataset = WebDataset("validation.tar").decode("pil").to_tuple("jpg;png", "json")
    
    return training_dataset, validation_dataset, vocab
