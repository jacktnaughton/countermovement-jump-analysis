import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.model import Autoencoder
from src.config import config
from src.data_loader import load_data

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data = load_data().to(device)

train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
train_data = train_data.to(device)
test_data = test_data.to(device)

# Create dataset and dataloader
dataset = TensorDataset(train_data, train_data)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model, optimizer, and loss function
autoencoder = Autoencoder(config["latent_dim"]).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
loss_function = nn.MSELoss()

# Training loop
for epoch in range(config["epochs"]):
    autoencoder.train()
    
    for x, _ in dataloader:
        optimizer.zero_grad()
        outputs = autoencoder(x)
        loss = loss_function(outputs, x)
        loss.backward()
        optimizer.step()

    # Evaluate model on test set
    autoencoder.eval()
    with torch.no_grad():
        test_outputs = autoencoder(test_data)
        test_loss = loss_function(test_outputs, test_data)

    if epoch % 1 == 0:
        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Save model
torch.save({
    'model_state_dict': autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, config["model_path"])

print(f"Model saved to {config['model_path']}")
