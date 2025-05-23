# scripts/data_cleaning.py
- How to run:
     - scripts/clean_data.py
- Prepare raw force-time series data from countermovement jumps for training VAE model
- What it does:
    - Loads:
        - jump_meta_data.csv -> athlete info
        - jump_ts_data.npy -> raw force-time series array
    - Preprocesses:
        - Replaces NaN values with 0s in time series
        - Computes total vertical force by summing across force plate columns.
        - Normalizes each time series by the initial force value, approximating body weight
    - Downsamples each jump:
        - Trims the series after the last active movement.
        - Interpolates each trimmed force curve to a uniform length (target_length = 100)
    - Outputs:
        - A cleaned DataFrame saved as a .pkl file in ../data/


# scripts/outlier_removal.py
- Loads: 
    - Prepocessed jump data
    - Trained autoencoder model from ../saved_models/autoencoder_stochastic.pth
- Latent embedding:
    - Computes the latent representation (encode) of each jump
- Reconstruction Loss:
    - Flags outliers using:
        - High latent inconsistency
        - Excessive reconstruction loss
        - Unrealistic jump height (e.g. > 40 inches)
        - High mean force output. 
- Saves:
    - outlier_removed.pkl: cleaned dataset
    - outliers.pkl: flagged jumps


# scripts/train.py
- How to run:
    - (From scripts folder)
    - ./train.sh
    - Ensure that the data source is defined inside of src/data_loader.py
    * Currently using cleaned data
- This script trains a Variational Autoencoder (VAE) on preprocessed jump data. It uses PyTorch for modeling, and the configuration is loaded from src/config.py. The trained model is saved as a .pth file in the specified in config["model_path"] path.
* Note, only VAE training block enabled, can uncomment Autoencoder block to train it instead or alongside VAE. 



# week_5_analysis.ipynb
- Saves new files binned_males and binned_females, containing only male/female athletes, with added columns [jump_start, jump_end, jump_duration_s, jump_end_time, Gender, takeoff_velocity, takeoff_velocity_2, tv_jump_height, tv_jump_height_2, vae_latent_space_4d, reconstructed_waveform, adjusted_reconstruction, predicted_takeoff_velocity, latent_predicted_jump_height, avg_diff_recon_vs_downsampled, avg_magnitude_downsampled, diff_ratio, jump_bin, t_n, latent_space_5d, latent_distance, latent_distance_orig]



# IF USING PROVIDED DATA ALREADY (don't need to run data_cleaning or outlier_removal/don't need to train model)
- to load a saved model state (model name contains date it was trained):
    - saved_models/
        - copy the saved name
        - Ex in the python file you want to load it into:
            vae_model_path = "../saved_models/vae20250408_230217_vae.pth"
            vae = VAE(latent_dim=latent_dim).to(device)  
            checkpoint = torch.load(vae_model_path, map_location=device)
            vae.load_state_dict(checkpoint['model_state_dict'])
            vae.eval()   
        
- in data/
    - binned_females.pkl/binned_males.pkl now contain:
        - jump_start/jump_end -> when the movement started/ended
        - jump_duration_s -> how long the movement from start to stop was in seconds
        - Gender -> athlete gender
        - takeoff_velocity -> calculated from downsampled time series
        - takeoff_velocity_2 -> exact same value as takeoff_velocity (was done for testing)
        - tv_jump_height -> jump height predicted from takeoff_velocity value
        - vae_latent_space_4d -> the 4d latent value that represents each jump (can be used in the decode function to reconstruct original jump)
        - adjusted_reconstruction -> contains adjusted values to account for 20 Newton difference for measuring the start of movement
        - latent_predicted_jump_height -> height predicted from latent reconstruction and jump_duration_s
        - jump_bin (str) -> whether the athlete falls in the bin of 6-9 in. jumps, 9-12 etc
        - latent_space_5d -> latent space with normalized time as an extra feature


# TO RUN THE PRESCRIBE_JUMP FUNCTION
- functions/prescibe_jump.py
    - ensure you have a: 
        - profileID: string containing profileID value of player you want to look at
        - model_df: a dataframe with jump data, i.e. binned_males or binned_females
        - variational autoencoder: using instructions above to load in VAE
        - top_n: number of jumps to average, default 1
        - delta: minimum jump height increase (i.e. increase the jump by 2 inches if delta = 2.0, default of 1.0)
        

# Jump Consistency and Optimization with Deep Learning

- This project uses deep learning to analyze the evolution and optimization of countermovement jump performance in athletes. By modeling jump force curves with a variational autoencoder (VAE), we extract compact latent space representations that capture essential biomechanical patterns.

# Project Overview

- We analyze athlete performance by tracking the consistency and progression of jump technique across multiple years of training. For each athlete, we compute the Euclidean distance between each jump’s latent vector and their annual average vector to quantify consistency over time.

# Key Methods and Contributions

- Latent Space Modeling: Jump force-time curves are encoded into a 4D latent space using a VAE, trained to reconstruct normalized curves.

- Consistency Metrics: For each athlete, we compute per-year latent centroids and measure how each jump deviates from the centroid. This quantifies temporal consistency in performance.

- Jump Improvement Prescription: A function (prescribe_jump_improvement) modifies an athlete’s latent vector minimally to simulate a jump with at least +1 inch in height, then resynthesizes the improved force curve.

- Inter-Sport Comparisons: By comparing latent space distributions across sports like Beach Volleyball and Acrobatics, we investigate domain-specific movement patterns and potential predictors of elite performance.

- High vs Low Performers: We contrast latent space patterns between top- and bottom-performing athletes within each sport to identify differentiating features.

This research enables data-driven feedback on jump technique evolution, performance optimization, and personalized training prescriptions for athletic development.


# Project Structure

project_root/
│── data/                     # processed datasets
│   ├── binned_females.pkl    # all male athletes, cleaned dataset
│   ├── binned_females.pkl    # all female athletes, cleaned dataset
│
│── functions/                # useful functions
│   ├── prescribe_jump.py     # used to prescribe jump improvements
│
│── saved_models/             # Trained models
│   ├── autoencoder_stochastic_{date}.pth
│
│── scripts/                  # Core scripts
│   ├── data_cleaning.py      # Preprocess raw jump data
│   ├── outlier_removal.py    # Detect and remove outliers using autoencoder
│   ├── train.py              # train model
│   ├── week_10.ipynb         # analysis containing reconstructions by top and bottom 10%
│
│── src/                      # Jupyter notebooks for interactive analysis
│   ├── \__init__.py
│   ├── config.py             # Contains configurations for model parameters and file paths
│   ├── data_loader.py        # used in model training to load the data
│   ├── model.py              # contains the structure for Autoencoder and Variational Autoencoder models
│
│── README.md                 # This file
│── Makefile (future)         # Shortcut commands (planned )




# Simplified Instructions
1. Train the Autoencoder

bash scripts/train.sh

2. Evaluate the Model

python scripts/evaluate.py

3. Preprocess Data

python scripts/data_cleaning.py

4. Remove Outliers

python scripts/outlier_removal.py


NOTES:

src/config.py contains batch_size, learning rate, etc



