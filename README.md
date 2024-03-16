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
[python script](data_preprocessing.py)<BR>
[documentation](docs/data_preprocessing.md)<br>
Training a predictive model requires preprocessing of data. This involves preparing a dataset with a question and the corresponding answer for the model to learn from. The type of dataset needed depends on the specific use case of the model. In this case, the model is an image captioning model, which requires a set of images and their corresponding descriptions.

The `data_preprocessing.py` script contains several functions and classes for preprocessing the data:

- `Vocabulary`: A class for mapping between words and numerical indices. This is used to convert the words in the captions into a format that can be processed by the model.

- `build_vocab`: A function that builds the vocabulary by tokenizing the captions and counting the frequency of each word. Words that appear less than a certain threshold are excluded from the vocabulary.

- `resize_image` and `preprocess_image`: Functions for resizing and preprocessing the images. The preprocessing involves resizing the image, randomly cropping it to a certain size, flipping it horizontally, converting it to a tensor, and normalizing it.

- `CaptionDataset`: A class for the caption dataset. This class loads the images and captions, preprocesses the images, tokenizes the captions, and converts the captions into numerical format using the vocabulary.

- `collate_fn`: A function for batching the data. This function sorts the data by caption length in descending order, stacks the images into a single tensor, and pads the captions so that they all have the same length.

The script also includes code for loading the annotations file, extracting the captions, creating the vocabulary, creating the datasets, splitting the training dataset into training and test sets, creating a subset of the training dataset for hyperparameter tuning, creating the data loaders, and logging the GPU memory usage.

## Usage

To train the model, run `python main.py`.

To use the trained model for inference, run `python inference.py`.

