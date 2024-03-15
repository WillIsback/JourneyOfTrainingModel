import torch
from data_preprocessing import test_loader, vocab
from model import CNN_to_RNN
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.use('Agg')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def caption_image(image_path, model, max_length=50):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Prepare an image
    print("\n" + "="*50)
    print("Loading and transforming image...")
    with tqdm(total=1, desc="Loading") as pbar:
        image = load_image(image_path, transform)
        pbar.update()
    
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    print("\n" + "="*50)
    print("Generating caption...")
    with tqdm(total=2, desc="Generating") as pbar:
        feature = model.encoder(image_tensor)
        pbar.update()
        sampled_ids = model.decoder.sample(feature)
        pbar.update()

    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    print("\n" + "="*50)
    print("Converting word IDs to words...")
    sampled_caption = []
    for word_id in tqdm(sampled_ids, desc="Converting"):
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print("\n" + "="*50)
    print("Generated Caption:")
    print(sentence)
    image = Image.open(image_path)
    plt.imshow(np.asarray(image))
    plt.savefig('output.png')
    plt.show()


def TestModel(device, model, test_loader, vocab_size):
    # Test the model
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, captions, lengths in test_loader:  # Replace with your actual test data loader
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            outputs = model(images, captions)
            outputs = outputs.view(-1, vocab_size)
            outputs = outputs[:targets.size(0)]

            _, predicted = outputs.max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy}")
    
    
if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Load model and weights
    model_path = "model.pt"  # replace with your model path
    embed_size = 1024
    hidden_size = 1024
    vocab_size = len(vocab)
    num_layers = 1  # replace with your number of LSTM layers
    print("Loading model...")
    model = CNN_to_RNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")
    model.eval()

    # Image path
    image_path = "Images/Rev_PostApo_00026_.png"  # replace with your image path
    caption_image(image_path, model)