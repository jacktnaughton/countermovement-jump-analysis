import sys
import os
sys.path.append("../")
print(os.getcwd())
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
from datetime import datetime
from src.model import Autoencoder  

latent_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder(latent_dim).to(device)

model_path = "../saved_models/autoencoder_stochastic.pth"
checkpoint = torch.load(model_path, map_location=device)

autoencoder.load_state_dict(checkpoint['model_state_dict'])

data = pd.read_pickle("../data/outlier_removed.pkl")  
sampled_df = data.sample(n=4000, random_state=42)

jump_features = ["JUMP_HEIGHT_INCHES"]
scaler = StandardScaler()
feature_matrix = scaler.fit_transform(sampled_df[jump_features])

# Stack latent space into a matrix
latent_matrix = np.stack(sampled_df["latent_space"].values)

# Concatenate latent space with the jump features
combined_matrix = np.hstack((latent_matrix, feature_matrix))

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
combined_tsne = tsne.fit_transform(combined_matrix)

sampled_df["tsne_combined_1"] = combined_tsne[:, 0]
sampled_df["tsne_combined_2"] = combined_tsne[:, 1]

sampled_df["initial_weight"] = sampled_df["total_force"].apply(lambda x: x[0] if isinstance(x, list) or isinstance(x, np.ndarray) else np.nan)

def get_mean_reconstructed_jump(region_df):
    """Computes the mean reconstructed jump for a given region."""
    reconstructed_jumps = np.array([get_reconstructed_jump(row["latent_space"]) for _, row in region_df.iterrows()])
    return np.mean(reconstructed_jumps, axis=0)  # Compute mean across all jumps


def plot_mean_region_jumps(region_df, x_range, y_range, save_path):
    """Plots the mean reconstructed jump and highlights the selected t-SNE region, displaying mean weight."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    region_name = f"Region ({x_min}, {y_min}) to ({x_max}, {y_max})"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  

    # Compute and plot the mean reconstructed jump
    mean_jump = get_mean_reconstructed_jump(region_df)
    axes[0].plot(mean_jump, linewidth=2.5, color="blue")
    
    # Compute the mean initial weight
    mean_weight = region_df["initial_weight"].mean()
    
    axes[0].set_title(f"Mean Reconstructed Jump for {region_name}\nMean Initial Weight: {mean_weight:.2f} lbs")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reconstructed Jump Height")
    axes[0].grid(True)

    # Scatter plot in t-SNE space colored by INITIAL_WEIGHT
    sns.scatterplot(x="tsne_combined_1", y="tsne_combined_2", 
                    data=sampled_df, alpha=0.7, hue="initial_weight", palette="viridis", ax=axes[1])

    # Highlight the selected region
    axes[1].plot([x_min, x_max, x_max, x_min, x_min], 
                 [y_min, y_min, y_max, y_max, y_min], 
                 color="red", linestyle="dashed", linewidth=3.5)  

    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].set_title(f"t-SNE Region: {region_name} (Colored by Initial Weight)")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save the plot
    plt.show()
    
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Function to get the reconstructed jump from the latent vector using the autoencoder
def get_reconstructed_jump(latent_vector):
    latent_tensor = torch.tensor(latent_vector, dtype=torch.float32)
    reconstructed_jump = autoencoder.decoder(latent_tensor)
    return reconstructed_jump.detach().numpy()


def get_regions(sampled_df):
    return [
        (sampled_df[
            (sampled_df["tsne_combined_1"] >= -60) & (sampled_df["tsne_combined_1"] <= -40) &
            (sampled_df["tsne_combined_2"] >= -40) & (sampled_df["tsne_combined_2"] <= -20)
        ], (-60, -40), (-40, -20), "Region 1"),
        
        (sampled_df[
            (sampled_df["tsne_combined_1"] >= 0) & (sampled_df["tsne_combined_1"] <= 20) &
            (sampled_df["tsne_combined_2"] >= 0) & (sampled_df["tsne_combined_2"] <= 20)
        ], (0, 20), (0, 20), "Region 2"),
        
        (sampled_df[
            (sampled_df["tsne_combined_1"] >= 60) & (sampled_df["tsne_combined_1"] <= 80) &
            (sampled_df["tsne_combined_2"] >= -20) & (sampled_df["tsne_combined_2"] <= 20)
        ], (60, 80), (-20, 20), "Region 3")
    ]

def get_mean_reconstructed_jump(region_df):
    """Computes the mean reconstructed jump for a given region."""
    reconstructed_jumps = np.array([get_reconstructed_jump(row["latent_space"]) for _, row in region_df.iterrows()])
    return np.mean(reconstructed_jumps, axis=0)  # Compute mean across all jumps

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
    axes[0].plot(mean_jump, linewidth=2.5, color="blue")
    axes[0].set_title(f"Mean Reconstructed Jump for {region_name}\nAvg Height: {mean_jump_height:.2f} inches")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reconstructed Jump Height")
    axes[0].grid(True)

    # Scatter plot for t-SNE space
    sns.scatterplot(x="tsne_combined_1", y="tsne_combined_2", 
                    data=sampled_df, alpha=0.7, hue="JUMP_HEIGHT_INCHES", palette="viridis", ax=axes[1])

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


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    output_folder = "../results"
    os.makedirs(output_folder, exist_ok=True)  
    
    for region_df, x_range, y_range, region_name in get_regions(sampled_df):
        save_path = f"{output_folder}/{region_name.replace(' ', '_').lower()}_mean_jump_{timestamp}.png"
        plot_region_with_mean_jump(region_df, x_range, y_range, region_name, save_path, sampled_df)
