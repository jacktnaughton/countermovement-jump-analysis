import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Add project root to path
sys.path.append("../")

from src.model import Autoencoder  # Import Autoencoder model

# ================================
# CONFIGURATION
# ================================
LATENT_DIM = 16  # Size of the latent space
MODEL_PATH = "../saved_models/autoencoder_stochastic.pth"
DATA_PATH = "../data/outlier_removed.pkl"
RESULTS_FOLDER = "../results"
N_SAMPLES = 4000
PERPLEXITY = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# LOAD MODEL
# ================================
autoencoder = Autoencoder(LATENT_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
autoencoder.load_state_dict(checkpoint["model_state_dict"])
autoencoder.eval()  # Set model to evaluation mode

# ================================
# LOAD DATA
# ================================
data = pd.read_pickle(DATA_PATH)
sampled_df = data.sample(n=N_SAMPLES, random_state=42)

# Normalize jump features
jump_features = ["JUMP_HEIGHT_INCHES"]
scaler = StandardScaler()
sampled_df[jump_features] = scaler.fit_transform(sampled_df[jump_features])

# Extract latent space and concatenate with jump height
latent_matrix = np.stack(sampled_df["latent_space"].values)
feature_matrix = scaler.transform(sampled_df[jump_features])
combined_matrix = np.hstack((latent_matrix, feature_matrix))

# ================================
# t-SNE TRANSFORMATION
# ================================
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42)
tsne_results = tsne.fit_transform(combined_matrix)

sampled_df["tsne_combined_1"] = tsne_results[:, 0]
sampled_df["tsne_combined_2"] = tsne_results[:, 1]

# Compute initial weight from force data
sampled_df["initial_weight"] = sampled_df["total_force"].apply(
    lambda x: x[0] if isinstance(x, (list, np.ndarray)) else np.nan
)

# ================================
# RECONSTRUCTION FUNCTION
# ================================
def get_reconstructed_jump(latent_vector):
    """Decodes a latent vector into a reconstructed jump."""
    latent_tensor = torch.tensor(latent_vector, dtype=torch.float32, device=DEVICE)
    reconstructed_jump = autoencoder.decoder(latent_tensor).cpu().detach().numpy()
    return reconstructed_jump

# ================================
# REGION SELECTION
# ================================
def get_regions(df):
    """Defines regions in the t-SNE space."""
    return [
        (df[
            (df["tsne_combined_1"] >= -60) & (df["tsne_combined_1"] <= -40) &
            (df["tsne_combined_2"] >= -40) & (df["tsne_combined_2"] <= -20)
        ], (-60, -40), (-40, -20), "Region 1"),
        
        (df[
            (df["tsne_combined_1"] >= 0) & (df["tsne_combined_1"] <= 20) &
            (df["tsne_combined_2"] >= 0) & (df["tsne_combined_2"] <= 20)
        ], (0, 20), (0, 20), "Region 2"),
        
        (df[
            (df["tsne_combined_1"] >= 60) & (df["tsne_combined_1"] <= 80) &
            (df["tsne_combined_2"] >= -20) & (df["tsne_combined_2"] <= 20)
        ], (60, 80), (-20, 20), "Region 3"),
    ]

# ================================
# MEAN RECONSTRUCTED JUMP
# ================================
def get_mean_reconstructed_jump(region_df):
    """Computes the mean reconstructed jump for a given region."""
    reconstructed_jumps = np.array([
        get_reconstructed_jump(row["latent_space"]) for _, row in region_df.iterrows()
    ])
    return np.mean(reconstructed_jumps, axis=0) if len(reconstructed_jumps) > 0 else None

# ================================
# PLOTTING FUNCTION
# ================================
def plot_region_with_mean_jump(region_df, x_range, y_range, region_name, save_path, sampled_df):
    """Plots the mean reconstructed jump and highlights the selected t-SNE region."""
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Compute the mean jump height
    mean_jump_height = region_df["JUMP_HEIGHT_INCHES"].mean()
    print(f"Average Jump Height for {region_name}: {mean_jump_height:.2f} inches")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Compute and plot the mean reconstructed jump
    mean_jump = get_mean_reconstructed_jump(region_df)
    if mean_jump is not None:
        axes[0].plot(mean_jump, linewidth=2.5, color="blue")
        axes[0].set_title(f"Mean Reconstructed Jump for {region_name}\nAvg Height: {mean_jump_height:.2f} inches")
    else:
        axes[0].set_title(f"No data in {region_name}")
    
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reconstructed Jump Height")
    axes[0].grid(True)

    # Scatter plot for t-SNE space
    sns.scatterplot(
        x="tsne_combined_1", y="tsne_combined_2", 
        data=sampled_df, alpha=0.7, hue="JUMP_HEIGHT_INCHES", palette="viridis", ax=axes[1]
    )

    # Highlight the selected region
    axes[1].plot([x_min, x_max, x_max, x_min, x_min], 
                 [y_min, y_min, y_max, y_max, y_min], 
                 color="red", linestyle="dashed", linewidth=3.5)

    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].set_title(f"t-SNE Region: {region_name} (Colored by Jump Height)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ================================
# EXECUTION
# ================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for region_df, x_range, y_range, region_name in get_regions(sampled_df):
        save_path = f"{RESULTS_FOLDER}/{region_name.replace(' ', '_').lower()}_mean_jump_{timestamp}.png"
        plot_region_with_mean_jump(region_df, x_range, y_range, region_name, save_path, sampled_df)
