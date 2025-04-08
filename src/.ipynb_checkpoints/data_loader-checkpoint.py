import pandas as pd
import torch

def load_data():
    # loads cleaned data from data folder
    df = pd.read_pickle('../data/cleaned_data.pkl')
    data = df['downsampled_time_series'].tolist()  
    data = torch.tensor(data, dtype=torch.float32)

    return data
