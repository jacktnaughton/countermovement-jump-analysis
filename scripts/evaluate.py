import torch
from src.model import Autoencoder
from src.config import config
from src.data_loader import load_data

# Load test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = load_data().to(device)

# Load model
checkpoint = torch.load(config["model_path"])
autoencoder = Autoencoder(config["latent_dim"]).to(device)
autoencoder.load_state_dict(checkpoint["model_state_dict"])
autoencoder.eval()

# Evaluate model
with torch.no_grad():
    reconstructed = autoencoder(test_data)
    loss_function = torch.nn.MSELoss()
    test_loss = loss_function(reconstructed, test_data)

print(f"Test Loss: {test_loss.item():.4f}")
