Jump Consistency Analysis

This project analyzes the consistency of countermovement jumps using deep learning techniques. The goal is to measure how an athlete's jump patterns change over years of training by analyzing latent space representations.

Project Overview

We use autoencoders to extract latent space representations of jump force curves and compute consistency metrics based on the Euclidean distance between jumps and yearly mean vectors. This helps track how an athlete's jumping technique evolves with experience.

project_root/
│── data/                     # Raw & processed datasets
│   ├── jump_meta_data.csv
│   ├── jump_ts_data.npy
│   ├── cleaned_data.pkl
│   ├── outlier_removed.pkl
│
│── results/                  # Output graphs and reports
│   ├── latent_space_consistency.png
│
│── saved_models/             # Trained models
│   ├── autoencoder_stochastic_{date}.pth
│
│── scripts/                  # Core scripts
│   ├── data_cleaning.py       # Preprocess raw jump data
│   ├── outlier_removal.py     # Detect and remove outliers using autoencoder
│   ├── analysis.py            # Compute jump consistency analysis
│
│── notebooks/                # Jupyter notebooks for interactive analysis
│   ├── exploratory_analysis.ipynb
│
│── README.md                 # This file
│── Makefile (future)         # Shortcut commands (planned )




1. Train the Autoencoder

bash scripts/train.sh

2. Evaluate the Model

python scripts/evaluate.py

3. Preprocess Data

python scripts/data_cleaning.py

4. Remove Outliers

python scripts/outlier_removal.py

5. Run the Analysis

python scripts/analysis.py

NOTES:

src/config.py contains batch_size, learning rate, etc