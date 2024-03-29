# model.py
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class RNN_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNN_Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None, max_len=20, top_k=5):
        """Generate captions for given image features using top-k sampling."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get top k probabilities and their indices
            top_probabilities, top_indices = probabilities.topk(top_k, dim=1)
            # Sample from the top k probabilities to get the next word's index
            sampled_index = torch.multinomial(top_probabilities, num_samples=1)
            # Get the corresponding word
            predicted = top_indices.gather(1, sampled_index)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class CNN_to_RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNN_to_RNN, self).__init__()
        self.encoder = CNN_Encoder(embed_size)
        self.decoder = RNN_Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs