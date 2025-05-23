import sys
sys.path.append("../")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.model import Autoencoder, VAE
from src.config import config
from src.data_loader import load_data
import pdb
# ===========================
# Device setup and data load
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data = load_data().to(device)

train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

dataset = TensorDataset(train_data, train_data)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Handle file paths robustly
base_path, _ = os.path.splitext(config["model_path"])
ae_path = f"{base_path}_autoencoder.pth"
vae_path = f"{base_path}_vae.pth"

# ======================
#    Autoencoder Block
# ======================
# autoencoder = Autoencoder(config["latent_dim"]).to(device)
# ae_optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
# ae_loss_fn = nn.MSELoss()

# for epoch in range(config["epochs"]):
#     autoencoder.train()
#     for x, _ in dataloader:
#         ae_optimizer.zero_grad()
#         outputs = autoencoder(x)
#         loss = ae_loss_fn(outputs, x)
#         loss.backward()
#         ae_optimizer.step()

#     autoencoder.eval()
#     with torch.no_grad():
#         test_outputs = autoencoder(test_data)
#         test_loss = ae_loss_fn(test_outputs, test_data)

#     if epoch % 10 == 0:
#         print(f"[AE] Epoch [{epoch+1}/{config['epochs']}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# torch.save({
#     'model_state_dict': autoencoder.state_dict(),
#     'optimizer_state_dict': ae_optimizer.state_dict()
# }, ae_path)
# print(f"Autoencoder model saved to {ae_path}")


# ======================
#        VAE Block
# ======================
vae = VAE(input_dim=100, latent_dim=config["latent_dim"]).to(device)
vae_optimizer = optim.Adam(vae.parameters(), lr=config["learning_rate"])

def vae_loss(x, x_hat, mean, logvar, beta=0.01):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    # print(f"recon loss: {recon_loss}, kl loss: {kl_div}")
    return recon_loss + beta * kl_div

for epoch in range(config["epochs"]):
    vae.train()
    for x, _ in dataloader:
        vae_optimizer.zero_grad()
        x_hat, mean, logvar = vae(x)
        loss = vae_loss(x, x_hat, mean, logvar)
        loss.backward()
        vae_optimizer.step()

    vae.eval()
    with torch.no_grad():
        x_hat, mean, logvar = vae(test_data)
        test_loss = vae_loss(test_data, x_hat, mean, logvar)

    if epoch % 10 == 0:
        print(logvar.mean())
        print(f"[VAE] Epoch [{epoch+1}/{config['epochs']}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

torch.save({
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': vae_optimizer.state_dict()
}, vae_path)
print(f"VAE model saved to {vae_path}")
