import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import random
import math

# Step 1: Data Preparation (Assuming you have tokenized_data)
import numpy as np
from nltk.tokenize import word_tokenize  # Assurez-vous d'avoir installé NLTK via pip
import matplotlib.pyplot as plt
from tqdm import tqdm


PATH="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.test.tok"
PATH="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.train.tok"

def openfile(file: str) -> list[str]:
    """
    prend un chemin ou une liste de chemain vers un ou des fichier(s) texte(s) 
    et renvoie la liste de mots que contients ce(s) texte(s)
    """
    text = []
    print(file)
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            if line[-1] == '\n':
                line = line[:-2]
            line = line.split(' ')

            text += line[2 : -2]

    return text

texte=openfile(PATH)

# Step 2: Build Vocabulary
word_counts = Counter()
for sentence in texte:
    word_counts.update(sentence)

vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
idx_to_word = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)

# Step 3: Subsampling of Frequent Words
def subsample_frequent_words(data, threshold=1e-5):
    total_words = sum(word_counts.values())
    word_freqs = {word: count / total_words for word, count in word_counts.items()}
    subsampled_data = []
    for sentence in data:
        subsampled_sentence = [word for word in sentence if random.uniform(0, 1) > (1 - math.sqrt(threshold / word_freqs[word]))]
        if len(subsampled_sentence) > 1:
            subsampled_data.append(subsampled_sentence)
    return subsampled_data

tokenized_data = subsample_frequent_words(tokenized_data)

# Step 4: Create Skip-gram Dataset
class SkipGramDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.samples = self._generate_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _generate_samples(self):
        samples = []
        for sentence in self.data:
            for i, target_word in enumerate(sentence):
                start = max(0, i - self.window_size)
                end = min(len(sentence), i + self.window_size + 1)
                for j in range(start, end):
                    if j != i:
                        samples.append((vocab[target_word], vocab[sentence[j]]))
        return samples

# Step 5: Define Hierarchical Softmax
class HierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(HierarchicalSoftmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tree = {}  # Construct your hierarchical softmax tree here

    def forward(self, target):
        # Implement hierarchical softmax here
        pass

# Step 6: Define Negative Sampling
class NegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_neg_samples):
        super(NegativeSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_neg_samples = num_neg_samples
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context, negative_samples):
        target_embeds = self.embeddings(target)
        context_embeds = self.embeddings(context)
        negative_embeds = self.embeddings(negative_samples)
        return target_embeds, context_embeds, negative_embeds

# Step 7: Define Skip-gram Model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_neg_samples):
        super(SkipGram, self).__init__()
        self.hierarchical_softmax = HierarchicalSoftmax(vocab_size, embedding_dim)
        self.negative_sampling = NegativeSampling(vocab_size, embedding_dim, num_neg_samples)

    def forward(self, target, context, negative_samples):
        # Implement Skip-gram with hierarchical softmax or negative sampling here
        pass

# Step 8: Define Training Loop
def train_skipgram(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for target, context, negative_samples in dataloader:
            target = target.to(device)
            context = context.to(device)
            negative_samples = negative_samples.to(device)

            optimizer.zero_grad()

            target_embeds, context_embeds, negative_embeds = model(target, context, negative_samples)

            # Calculate your loss here (either hierarchical softmax or negative sampling)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

# Step 9: Hyperparameters
embedding_dim = 100
window_size = 5
learning_rate = 0.01
batch_size = 64
num_epochs = 10
num_neg_samples = 10  # Number of negative samples for Negative Sampling

# Step 10: Create DataLoader
skipgram_dataset = SkipGramDataset(tokenized_data, window_size)
dataloader = DataLoader(skipgram_dataset, batch_size=batch_size, shuffle=True)

# Step 11: Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGram(vocab_size, embedding_dim, num_neg_samples).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Step 12: Train the Model
train_skipgram(model, dataloader, optimizer, num_epochs)

# Step 13: Get Word Embeddings
# Extract word embeddings from the model's embeddings layer.

# Example: Get the embedding for a specific word
word = "word_to_lookup"
word_idx = vocab[word]
word_embedding = model.negative_sampling.embeddings(torch.LongTensor([word_idx]).to(device)).squeeze().detach().cpu().numpy()

# Step 14: Save the Model and Embeddings
# You can save the trained model and embeddings for later use.
