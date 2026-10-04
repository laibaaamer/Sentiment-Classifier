import os
import re
import random
import urllib.request
import zipfile
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE INFORMATION")
print("Using Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
DATA_URL = ("https://ai.stanford.edu/~amaas/data/sentiment/"
            "aclImdb_v1.tar.gz")
DATA_FILE = "aclImdb_v1.tar.gz"
DATA_DIR = "aclImdb"
if not os.path.exists(DATA_DIR):
    print("Downloading IMDb dataset...")
    urllib.request.urlretrieve(DATA_URL,DATA_FILE)
    print("Extracting IMDb dataset...")
    import tarfile
    with tarfile.open(DATA_FILE, "r:gz") as tar:
        tar.extractall()
    print("IMDb dataset ready!")
def load_reviews(folder):
    texts = []
    labels = []
    for label_name, label in [("pos", 1),("neg", 0)]:
        path = os.path.join(folder,label_name)
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                file_path = os.path.join(path,filename)
                with open(file_path,"r",encoding="utf-8") as file:
                    text = file.read()
                texts.append(text)
                labels.append(label)
    return texts, labels
train_path = os.path.join(DATA_DIR,"train")
test_path = os.path.join(DATA_DIR,"test")
train_texts, train_labels = load_reviews(train_path)
test_texts, test_labels = load_reviews(test_path)
print("\nTraining Samples:", len(train_texts))
print("Testing Samples :", len(test_texts))
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:['’][a-z]+)*|[!?]+",text)
    return tokens
MAX_VOCAB_SIZE = 20000
counter = Counter()
print("Building vocabulary...")
for text in train_texts:
    tokens = tokenize(text)
    counter.update(tokens)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
vocab = {PAD_TOKEN: 0,UNK_TOKEN: 1}
for word, frequency in counter.most_common(MAX_VOCAB_SIZE):
    if word not in vocab:
        vocab[word] = len(vocab)
print("Vocabulary Size:", len(vocab))
def text_to_sequence(text):
    tokens = tokenize(text)
    sequence = [vocab.get(token,vocab[UNK_TOKEN])for token in tokens]
    return sequence
MAX_LENGTH = 200
def pad_sequence(sequence):
    sequence = sequence[:MAX_LENGTH]
    if len(sequence) < MAX_LENGTH:
        sequence = sequence + [vocab[PAD_TOKEN]] * (MAX_LENGTH - len(sequence))
    return sequence
class IMDbDataset(Dataset):
    def __init__(self,texts,labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        sequence = text_to_sequence(self.texts[index])
        sequence = pad_sequence(sequence)
        sequence = torch.tensor(sequence,dtype=torch.long)
        label = torch.tensor(self.labels[index],dtype=torch.float)
        return sequence, label
full_train_dataset = IMDbDataset(train_texts,train_labels)
train_size = int(0.90 * len(full_train_dataset))
validation_size = (len(full_train_dataset)- train_size)
train_dataset, validation_dataset = random_split(full_train_dataset,[train_size, validation_size],generator=torch.Generator().manual_seed(SEED))
test_dataset = IMDbDataset(test_texts,test_labels)
print("Dataset Split")
print("Training   :", len(train_dataset))
print("Validation :", len(validation_dataset))
print("Testing    :", len(test_dataset))
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=torch.cuda.is_available())
validation_loader = DataLoader(validation_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
GLOVE_URL = ("https://nlp.stanford.edu/data/glove.6B.zip")
GLOVE_ZIP = "glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
if not os.path.exists(GLOVE_FILE):
    if not os.path.exists(GLOVE_ZIP):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(GLOVE_URL,GLOVE_ZIP)
        print("GloVe download complete!")
    print("Extracting GloVe...")
    with zipfile.ZipFile(GLOVE_ZIP,"r") as zip_ref:
        zip_ref.extract(GLOVE_FILE,".")
    print("GloVe ready!")
EMBEDDING_DIM = 100
def load_glove_embeddings(glove_file,vocab):
    print("Loading GloVe embeddings...")
    embedding_matrix = np.random.normal(scale=0.6,size=(len(vocab),EMBEDDING_DIM)).astype(np.float32)
    embedding_matrix[vocab[PAD_TOKEN]] = np.zeros(EMBEDDING_DIM)
    found = 0
    with open(glove_file,"r",encoding="utf-8") as file:
        for line in file:
            values = line.rstrip().split()
            word = values[0]
            if word in vocab:
                vector = np.asarray(values[1:],dtype=np.float32)
                embedding_matrix[vocab[word]] = vector
                found += 1
    print(f"Found GloVe vectors for {found}/{len(vocab)} vocabulary words")
    return torch.tensor(embedding_matrix,dtype=torch.float)
pretrained_embeddings = load_glove_embeddings(GLOVE_FILE,vocab)
class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim,1)
    def forward(self,lstm_output,mask):
        scores = self.attention(lstm_output).squeeze(-1)
        scores = scores.masked_fill(mask == 0,-1e9)
        attention_weights = torch.softmax(scores,dim=1)
        context = torch.sum(lstm_output* attention_weights.unsqueeze(-1),dim=1)
        return context
class SentimentBiLSTMAttention(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_layers=2,dropout=0.5,pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=vocab[PAD_TOKEN])
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout)
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2,1)
    def forward(self, x):
        mask = (x != vocab[PAD_TOKEN]).long()
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        context = self.attention(lstm_output,mask)
        context = self.dropout(context)
        output = self.fc(context)
        return output.squeeze(1)
VOCAB_SIZE = len(vocab)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5
model = SentimentBiLSTMAttention(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pretrained_embeddings=pretrained_embeddings)
model = model.to(device)
print("MODEL")
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=1)
def train_model(model,loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for texts, labels in loader:
        texts = texts.to(device,non_blocking=True)
        labels = labels.to(device,non_blocking=True)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        predicted = (torch.sigmoid(predictions) >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return (total_loss / len(loader),accuracy)
def evaluate_model(model,loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            predictions = model(texts)
            loss = criterion(predictions,labels)
            total_loss += loss.item()
            predicted = (torch.sigmoid(predictions) >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return (total_loss / len(loader),accuracy)
EPOCHS = 10
train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []
best_validation_accuracy = 0.0
print("TRAINING")
for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_model(model,train_loader)
    validation_loss, validation_accuracy = evaluate_model(model,validation_loader)
    scheduler.step(validation_loss)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = (validation_accuracy)
        torch.save(model.state_dict(),"best_sentiment_lstm.pth")
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.4f}")
    print(f"Validation Loss: "
        f"{validation_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
model.load_state_dict(torch.load("best_sentiment_lstm.pth",map_location=device))
test_loss, test_accuracy = evaluate_model(model,test_loader)
print("FINAL TEST RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
epochs_range = range(1,EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range,train_losses,marker="o",label="Training Loss")
plt.plot(epochs_range,validation_losses,marker="o",label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png",dpi=300)
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(epochs_range,train_accuracies,marker="o",label="Training Accuracy")
plt.plot(epochs_range,validation_accuracies,marker="o",label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png",dpi=300)
plt.show()
torch.save(model.state_dict(),"sentiment_bilstm_attention.pth")
print("Model saved successfully!")
def predict_sentiment(text):
    model.eval()
    sequence = text_to_sequence(text)
    sequence = pad_sequence(sequence)
    tensor = torch.tensor(sequence,dtype=torch.long).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output).item()
    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability
    return (sentiment,probability,confidence)
reviews = [
    "This movie was amazing and I really loved it.",
    "The movie was boring and disappointing.",
    "Excellent acting and a wonderful story.",
    "I hated this movie. It was terrible.",
    "I wasn't expecting much, but this movie was fantastic!",
    "This movie wasn't good at all and I didn't enjoy it."]
print("SAMPLE PREDICTIONS")
for review in reviews:
    sentiment, probability, confidence = (predict_sentiment(review))
    print("Review:", review)
    print("Sentiment:",sentiment)
    print(f"Probability: {probability:.4f}")
    print(f"Confidence: {confidence:.4f}")