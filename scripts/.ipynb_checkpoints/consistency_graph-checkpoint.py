import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

print("Loading processed data...")
data = pd.read_pickle("../data/outlier_removed.pkl")  
print(data)

# years training computed by taking first year value, and computing each year after (i.e. 2020 = year 0, even if December 2020)
print("Computing training years...")
data['recordedUTC'] = pd.to_datetime(data['recordedUTC'])
data['start_year'] = data.groupby('profileID')['recordedUTC'].transform('min').dt.year
data['years_training'] = data['recordedUTC'].dt.year - data['start_year']
data['latent_space'] = data['latent_space'].apply(np.array)

# latent space differences
print("Computing latent space differences...")

differences_per_year = []

for profile_id, profile_data in data.groupby('profileID'):
    for year, year_data in profile_data.groupby('years_training'):
        
        latent_vectors = torch.tensor(np.stack(year_data['latent_space'].values), dtype=torch.float32)
        
        # Mean latent space vector for this year
        mean_latent = latent_vectors.mean(dim=0)
        
        # Mean difference between each rep and the yearly mean
        differences = torch.norm(latent_vectors - mean_latent, dim=1).numpy()  # Euclidean distance
        
        for diff, (_, row) in zip(differences, year_data.iterrows()):
            differences_per_year.append({
                'profileID': profile_id, 
                'years_training': year, 
                'difference': diff, 
                'outlier': row['outlier']  
            })

diff_df = pd.DataFrame(differences_per_year)

filtered_diff_df = diff_df[diff_df["outlier"] == 0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Get current date-time


# plots
print("Generating plot...")
plt.figure(figsize=(8, 12))  
ax1 = plt.subplot(211)  
filtered_diff_df.boxplot(column='difference', by='years_training', ax=ax1)
ax1.set_ylim(0, 1.5)  
ax1.set_title(f"Mean Latent Space Differences by Years of Training (Outliers Removed) {timestamp}")
ax1.set_xlabel("Years of Training")
ax1.set_ylabel("Mean Latent Space Difference")
ax1.grid(True)

plt.tight_layout()

# save to results folder
output_folder = "../results"
os.makedirs(output_folder, exist_ok=True)  

output_path = f"{output_folder}/latent_space_consistency_{timestamp}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to {output_path}")
plt.show()
