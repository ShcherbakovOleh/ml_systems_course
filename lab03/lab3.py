import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load and prepare Fashion-MNIST (Mean=0.2860, Std=0.3530)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split official train (60k) into 55k train and 5k validation
train_dataset, val_dataset = random_split(full_train_dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 2. Define three models (target range: 100k–150k parameters)

# --- MLP ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# --- CNN ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# --- Vision Transformer (Attention) ---
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_dim = 1 * 7 * 7  # 49
        self.d_model = 64

        self.projection = nn.Linear(self.patch_dim, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B = x.shape[0]
        # Split 28x28 image into 16 patches of 7x7
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(B, 1, 4, 4, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, 16, -1)
        
        x = self.projection(patches)
        x = self.transformer(x)
        x = x.mean(dim=1) # Mean pooling
        return self.fc(x)


models = {
    'MLP': MLP().to(device),
    'CNN': CNN().to(device),
    'Attention': VisionTransformer().to(device)
}


# 3. Define latency measurement function
def measure_latency(model, batch_size, runs=50):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
    
    with torch.no_grad():
        for _ in range(10): # Warm-up
            _ = model(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            
    return np.median(latencies)


# Profile models (Parameters, FLOPs, Latency)
for name, model in models.items():
    print(f"\n=== {name} ===")
    summary(model, input_size=(1, 1, 28, 28))
    
    dummy_in = torch.randn(1, 1, 28, 28).to(device)
    macs, params = profile(model, inputs=(dummy_in,), verbose=False)
    flops = macs * 2 # 1 MAC ≈ 2 FLOPs
    print(f"THOP Measured FLOPs (Batch 1): {flops:,}")
    
    lat_1 = measure_latency(model, batch_size=1, runs=50)
    lat_128 = measure_latency(model, batch_size=128, runs=20)
    print(f"Latency Batch 1: {lat_1:.3f} ms | Batch 128: {lat_128:.3f} ms")


# 4. Train models (5 epochs with Adam, lr=1e-3, batch=128)
def train_and_eval(model, train_loader, test_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

results = {}
params_cnt = []
latencies_b128 = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    acc = train_and_eval(model, train_loader, test_loader, epochs=5)
    results[name] = acc
    params_cnt.append(sum(p.numel() for p in model.parameters()))
    latencies_b128.append(measure_latency(model, 128, 20))
    print(f"{name} Test Accuracy: {acc:.2f}%")


# 5. Plot and save Trade-off figures
names = list(results.keys())
accuracies = [results[n] for n in names]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy vs Parameters
ax[0].scatter(params_cnt, accuracies, color='red', s=100)
for i, txt in enumerate(names):
    ax[0].annotate(txt, (params_cnt[i], accuracies[i]))
ax[0].set_xlabel('Parameter Count')
ax[0].set_ylabel('Test Accuracy (%)')
ax[0].set_title('Accuracy vs Parameters')

# Accuracy vs Latency (Batch 128)
ax[1].scatter(latencies_b128, accuracies, color='blue', s=100)
for i, txt in enumerate(names):
    ax[1].annotate(txt, (latencies_b128[i], accuracies[i]))
ax[1].set_xlabel('Inference Latency (ms, Batch 128)')
ax[1].set_ylabel('Test Accuracy (%)')
ax[1].set_title('Accuracy vs Latency')

plt.tight_layout()
plt.savefig('tradeoff_plots.png', dpi=300)
print("\nPlot successfully saved as 'tradeoff_plots.png'!")
plt.show()