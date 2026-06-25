import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.dataset import DenoiseDataset
from models.residual_cnn import ResidualCNN 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_NOISY = "C:\\Users\\Aksha\\Downloads\\NoisyLR"
TRAIN_GT = "C:\\Users\\Aksha\\Downloads\\GT"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

dataset = DenoiseDataset(TRAIN_NOISY, TRAIN_GT)
loader = DataLoader(dataset, batch_size=2, shuffle=True)  # batch 2 safer for 512

model = ResidualCNN().to(device)

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

print(f"Trainable Parameters: {count_parameters(model):,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50

best_loss = float("inf")
loss_history = []

for epoch in range(epochs):
    total_loss = 0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)
        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "results/checkpoints/residual_cnn.pth")
        print("Best model saved.")
    
    loss_history.append(avg_loss)


print("Training complete.")