import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import datetime

# time stamp for file naming
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cleaned_data_path = f"../data/cleaned_data_{current_time}.pkl"

data = pd.read_csv("../data/jump_meta_data.csv")
ts_data = np.load("../data/jump_ts_data.npy")

ts_data[np.isnan(ts_data)] = 0.0

data["time_series"] = list(ts_data)

# total force for each time series
data["total_force"] = data["time_series"].apply(lambda x: x.sum(axis=1))

# initial force value for normalization based on athletes initial 'weight'
data["initial_value"] = data["total_force"].apply(lambda x: x[0])

# normalize time series by dividing total time series by individuals initial first value
data["normalized_time_series"] = data["total_force"] / data["initial_value"]

# Convert to PyTorch tensor and put on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = torch.tensor(data["normalized_time_series"].tolist(), dtype=torch.float32).to(device)

def find_jump_end(sequence, threshold=1e-3):
    """Finds the last non-zero value before the flatline starts."""
    nonzero_indices = (sequence.abs() > threshold).nonzero(as_tuple=True)[0]
    return nonzero_indices[-1].item() if len(nonzero_indices) > 0 else len(sequence) - 1

def process_jump_data(data, target_length=100):
    """
    Normalizes and resamples jump data.

    Parameters:
        data (torch.Tensor): Original data of shape [num_samples, time_steps]
        target_length (int): Desired length for all jumps after resampling

    Returns:
        torch.Tensor: Processed data of shape [num_samples, target_length]
    """
    num_samples, original_length = data.shape
    processed_data = []

    for i in range(num_samples):
        seq = data[i]
        jump_end = find_jump_end(seq)

        # Trim and interpolate
        trimmed_seq = seq[:jump_end + 1].unsqueeze(0)  
        resampled_seq = F.interpolate(trimmed_seq.unsqueeze(0), size=target_length, mode='linear', align_corners=False).squeeze(0).squeeze(0)

        processed_data.append(resampled_seq)

    return torch.stack(processed_data)

target_length = 100  
processed_data = process_jump_data(test_data, target_length)

data["downsampled_time_series"] = processed_data.cpu().numpy().tolist()  
data.to_pickle(cleaned_data_path)

print(f"Data cleaning complete. Saved to {cleaned_data_path}")
