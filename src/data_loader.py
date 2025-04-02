import torch
import pandas as pd

def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    return torch.tensor(df.values, dtype=torch.float32)
