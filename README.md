# Image Captioning Model

This repository contains the implementation of an image captioning model using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) in an encoder-decoder architecture. The model is trained on the COCO dataset, and the weights of the pre-trained ResNet50 model used in the CNN Encoder are not updated during training. The RNN Decoder is trained from scratch.

<details>
<summary>1. Model Architecture</summary>

[Go to section](#model-architecture)

</details>

<details>
<summary>2. Training</summary>

[Go to section](#training)

</details>

<details>
<summary>3. Data Preprocessing</summary>

[Go to section](#data_preprocessing)

</details>

<details>
<summary>4. Inference</summary>

[Go to section](#inference)

</details>

<details>
<summary>5. Usage</summary>

[Go to section](#usage)

</details>

## Model Architecture

The model consists of two main parts:

1. **CNN Encoder**: This part of the model is responsible for extracting features from the input images. It uses a pre-trained ResNet50 model, which has been trained on the ImageNet dataset for the task of image classification. The ResNet50 model transforms the complex image data into a simpler representation that still contains the important information.

2. **RNN Decoder**: This part of the model is responsible for generating captions based on the features extracted by the CNN Encoder. It is a sequence generation model that generates a sequence of words (the caption) one at a time. The RNN Decoder is trained from scratch during the training process.

## Training

The model is trained on the COCO dataset. During training, the weights of the ResNet50 model are not updated. Only the weights of the RNN Decoder are updated.

The state dictionary of the entire model (including both the CNN Encoder and RNN Decoder) is saved whenever the validation loss decreases. This state dictionary can be loaded later to use the trained model for inference.

### Data_Preprocessing
[data_preprocessing](data_preprocessing.py)<BR>
Training a predictive model requires preprocessing of data. This involves preparing a dataset with a question and the corresponding answer for the model to learn from. The type of dataset needed depends on the specific use case of the model. In this case, the model is an image captioning model, which requires a set of images and their corresponding descriptions.

The `data_preprocessing.py` script contains several functions and classes for preprocessing the data:

- `Vocabulary`: A class for mapping between words and numerical indices. This is used to convert the words in the captions into a format that can be processed by the model.

- `build_vocab`: A function that builds the vocabulary by tokenizing the captions and counting the frequency of each word. Words that appear less than a certain threshold are excluded from the vocabulary.

- `resize_image` and `preprocess_image`: Functions for resizing and preprocessing the images. The preprocessing involves resizing the image, randomly cropping it to a certain size, flipping it horizontally, converting it to a tensor, and normalizing it.

- `CaptionDataset`: A class for the caption dataset. This class loads the images and captions, preprocesses the images, tokenizes the captions, and converts the captions into numerical format using the vocabulary.

- `collate_fn`: A function for batching the data. This function sorts the data by caption length in descending order, stacks the images into a single tensor, and pads the captions so that they all have the same length.

The script also includes code for loading the annotations file, extracting the captions, creating the vocabulary, creating the datasets, splitting the training dataset into training and test sets, creating a subset of the training dataset for hyperparameter tuning, creating the data loaders, and logging the GPU memory usage.

#### Words Preprocessing

The words in the captions are preprocessed by tokenizing them and converting them into numerical format. This is done using the `nltk.tokenize.word_tokenize` function and the `Vocabulary` class. The `build_vocab` function is used to build the vocabulary, which includes special tokens for padding (`<pad>`), the start of a sentence (`<start>`), the end of a sentence (`<end>`), and unknown words (`<unk>`).
#### Image Preprocessing

The images in the dataset are preprocessed using several steps. The `resize_image` function is used to resize the image to a specified size using the `PIL.Image.resize` method with the `ANTIALIAS` filter. The `preprocess_image` function is used to apply a series of transformations to the image:

1. Resizing the image to 256 pixels on the shortest side.
2. Randomly cropping a 224x224 patch from the image.
3. Randomly flipping the image horizontally.
4. Converting the image to a PyTorch tensor.
5. Normalizing the tensor with mean and standard deviation values of [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively, which are the values used by the pre-trained ResNet model.

These transformations are implemented using the `torchvision.transforms` module and are encapsulated in a `transforms.Compose` object.

#### Dataset Creation and Splitting

The `CaptionDataset` class is used to create a dataset of image-caption pairs. The `__getitem__` method of this class loads an image and its corresponding caption, preprocesses the image, tokenizes the caption, and converts the caption into numerical format using the vocabulary.

The `random_split` function from the `torch.utils.data` module is used to split the training dataset into a training set and a test set. The size of the test set is 20% of the size of the original training dataset.

A subset of the training dataset is also created for hyperparameter tuning. The size of this subset is the same as the size of the validation dataset or one-thousandth of the size of the training dataset, whichever is smaller.

#### Data Loading

The `DataLoader` class from the `torch.utils.data` module is used to create data loaders for the training set, validation set, test set, and subset. The data loaders shuffle the data, batch the data, and load the data in parallel using multiple workers.

The `collate_fn` function is used to batch the data. This function sorts the data by caption length in descending order, stacks the images into a single tensor, and pads the captions so that they all have the same length.

#### GPU Memory Usage Logging

The GPU memory usage is logged during the training process. The `get_gpu_memory` function from the `Utils.gpu_utils` module is used to get the current GPU memory usage. The memory usage is logged to a file named 'gpu_memory_usage.log' using the `logging` module. The memory usage is logged for each epoch of the training process.
## Inference

During inference, the trained model takes an image as input, extracts features from the image using the CNN Encoder, and then generates a caption for the image using the RNN Decoder.

## Usage

To train the model, run `python main.py`.

To use the trained model for inference, run `python inference.py`.

