import sys
import os
sys.path.append("../")
print(os.getcwd())
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from src.model import Autoencoder  


# load data
print("Loading cleaned data...")
data = pd.read_pickle("../data/cleaned_data.pkl")
processed_data = np.array(data["downsampled_time_series"].tolist())
processed_data = torch.tensor(processed_data, dtype=torch.float32)  


# load autoencoder params
print("Loading trained autoencoder...")
latent_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder(latent_dim).to(device)

model_path = "../saved_models/autoencoder_stochastic.pth"
checkpoint = torch.load(model_path, map_location=device)

autoencoder.load_state_dict(checkpoint['model_state_dict'])

autoencoder.eval()  
print(f"Model loaded from {model_path}")

# latent space computation
with torch.no_grad():
    latent_representations = autoencoder.encode(processed_data.to(device))

# define latent space
data["latent_space"] = latent_representations.cpu().numpy().tolist()

# compute consisteny (i.e. take mean of latent space per session, subtract each jump's latent dims from mean, take average of distances)
data['latent_space'] = data['latent_space'].apply(np.array)
sessions_and_profiles = data.groupby(['profileID', 'recordedUTC'])
consistency_measures = []

for (profile_id, session_time), session_data in sessions_and_profiles:
    session_latent_means = np.mean(np.stack(session_data['latent_space'].values), axis=0)  
    for latent_vector in session_data['latent_space']:
        diff = np.abs(latent_vector - session_latent_means)   
        mean_diff = np.mean(diff)    
        consistency_measures.append(mean_diff)

data['latent_space_consistency'] = consistency_measures

# reconstruction error
def compute_reconstruction(autoencoder, data):
    losses = []
    
    for _, row in data.iterrows():
        original = torch.tensor(row["downsampled_time_series"], dtype=torch.float32, device=device)
        latent = torch.tensor(row["latent_space"], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            reconstructed = autoencoder.decoder(latent)
        
        loss = F.mse_loss(reconstructed, original)
        losses.append(loss.item())
    
    return np.array(losses)


print("Computing reconstruction loss...")
reconstruction_losses = compute_reconstruction(autoencoder, data)
data["reconstruction_loss"] = reconstruction_losses

# set outliers
print("Identifying outliers...")

reconstruction_threshold = np.percentile(data["reconstruction_loss"], 99)
avg_threshold = np.percentile(data["downsampled_time_series"].apply(np.mean), 95)

data["outlier"] = (
    (data["latent_space_consistency"] > 5) | 
    (data["JUMP_HEIGHT_INCHES"] > 40) |  
    (data["reconstruction_loss"] > reconstruction_threshold) | 
    (data["downsampled_time_series"].apply(np.mean) > avg_threshold) 
).astype(int)

# add initial weight
data["initial_weight"] = data["total_force"].apply(lambda x: x[0] if isinstance(x, list) or isinstance(x, np.ndarray) else np.nan)

# save outliers and nonoutliers
filtered_data = data[data["outlier"] == 0].copy()
outliers = data[data["outlier"] == 1].copy()

print(filtered_data)

filtered_data.to_pickle("../data/outlier_removed.pkl")
outliers.to_pickle("../data/outliers.pkl")

print(f"Outlier removal complete. {len(outliers)} outliers removed.")
print("Filtered data saved as 'data/outlier_removed.pkl'")
print("Outlier data saved as 'data/outliers.pkl'")
