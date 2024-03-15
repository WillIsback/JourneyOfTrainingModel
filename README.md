# Image Captioning Model

This repository contains the implementation of an image captioning model. The model uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) in an encoder-decoder architecture.

## Model Architecture

The model consists of two main parts:

1. **CNN Encoder**: This part of the model is responsible for extracting features from the input images. It uses a pre-trained ResNet50 model, which has been trained on the ImageNet dataset for the task of image classification. The ResNet50 model transforms the complex image data into a simpler representation that still contains the important information.

2. **RNN Decoder**: This part of the model is responsible for generating captions based on the features extracted by the CNN Encoder. It is a sequence generation model that generates a sequence of words (the caption) one at a time. The RNN Decoder is trained from scratch during the training process.

## Training

The model is trained on the COCO dataset. During training, the weights of the ResNet50 model are not updated. Only the weights of the RNN Decoder are updated.

The state dictionary of the entire model (including both the CNN Encoder and RNN Decoder) is saved whenever the validation loss decreases. This state dictionary can be loaded later to use the trained model for inference.

## Inference

During inference, the trained model takes an image as input, extracts features from the image using the CNN Encoder, and then generates a caption for the image using the RNN Decoder.

## Usage

To train the model, run `python train.py`.

To use the trained model for inference, run `python inference.py`.