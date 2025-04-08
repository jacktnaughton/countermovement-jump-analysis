import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"../saved_models/vae{current_time}.pth"

config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 50,
    "latent_dim": 4,
    "model_path": model_path
}
