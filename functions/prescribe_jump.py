"""
Example usage:

    from prescribe_jump import prescribe_jump_improvement
    prescribe_jump_improvement(profileID="dd817868-71dc-4126-b6c0-dfbd1212d527", model_df=binned_males, top_n=1, vae=vae, delta=3)

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping
from scipy.stats import pearsonr

def resample_waveform(waveform, old_duration, new_duration, num_points=101):
    old_time = np.linspace(0, old_duration, len(waveform))
    new_time = np.linspace(0, new_duration, num_points)
    interpolator = interp1d(old_time, waveform, kind='linear', fill_value="extrapolate")
    return interpolator(new_time)

def prescribe_jump_improvement(profileID: str, model_df: pd.DataFrame, vae, top_n=1, delta=1.0):
    """
    Prescribes a modification to an athlete's latent vector to improve jump height
    using a trained VAE model and biomechanical prediction from reconstructed force curves.

    Parameters:
    -----------
    model : nn.Module
        Trained VAE model with encode/decode methods.
    profileID : str or int
        Unique identifier for the athlete.
    model_df : pd.DataFrame
        DataFrame containing latent vectors, and metadata for all jumps.
    jump_duration : float
        Duration of the jump in seconds (used for normalization).
    top_n : int, default=1
        Number of recent jumps to average for prescribing improvement.
    delta : float, default=0.025
        Minimum desired increase in jump height (in inches).

    Returns:
    --------
    None
        Displays plots comparing original vs prescribed jump force curves.
    """
    jump_mean = model_df["jump_duration_s"].mean()
    jump_std = model_df["jump_duration_s"].std()
    vae.eval()
    athlete_df = model_df[model_df["profileID"] == profileID].copy()
    
    if athlete_df.empty:
        print(f"No data found for profileID: {profileID}")
        return

    top_jumps = athlete_df.sort_values("recordedUTC", ascending=False).head(top_n)

    for idx, row in top_jumps.iterrows():
        latent_vec = np.array(row["vae_latent_space_4d"])
        jump_duration = row["jump_duration_s"]
        initial_val = row["initial_value"]
        jump_date = row["recordedUTC"]
        actual_recorded_height = row["JUMP_HEIGHT_INCHES"]

        if initial_val < 1e-3 or jump_duration > 2.0 or jump_duration <= 0:
            print(f"[SKIP] jump invalid — initial_val: {initial_val}, duration: {jump_duration:.3f}s")
            continue

        t_n = (jump_duration - jump_mean) / jump_std
        latent_5d = np.concatenate([latent_vec, [t_n]])
        print(f"The 5d Latent Space (with time): {latent_5d}\n")

        def predict_height(z):
            """
            Predict jump height given a latent vector by decoding, computing velocity, and using physics.
            """
            z_latent = z[:4]
            t_n_pred = z[4]
            jump_t = t_n_pred * jump_std + jump_mean

            if jump_t <= 0 or jump_t > 2.0:
                return -100.0  # invalid duration penalty

            with torch.no_grad():
                z_tensor = torch.tensor(z_latent, dtype=torch.float32).unsqueeze(0)
                rec = vae.decode(z_tensor).squeeze().cpu().numpy()

            rec = np.clip(rec, 0, 5)
            offset = rec[0] * (20 / initial_val)
            adjusted = rec - offset
            net_force = adjusted - 1.0
            avg_net_force = np.mean(net_force)
            takeoff_velocity = avg_net_force * jump_t * 9.81
            height = (takeoff_velocity ** 2) / (2 * 9.81) * 39.37
            return height

        predicted_orig_height = predict_height(latent_5d)
        target_jump_height = predicted_orig_height + delta

        def objective(z_new):
            """
            Optimization objective: stay close to original z but achieve at least (original + delta) height.
            """
            pred = predict_height(z_new)
            penalty = max(0, target_jump_height - pred) ** 2
            return np.sum((z_new - latent_5d) ** 2) + 10 * penalty

        epsilon = 0.05

        def constraint(z_new):
            """
            Constraint function to ensure new height is >= original height + delta.
            """
            return predict_height(z_new) - target_jump_height

        def min_latent_shift(z_new):
            """
            Constraint: enforce a minimum shift in latent space to avoid trivial solutions.
            """
            return np.linalg.norm(z_new - latent_5d) - epsilon

        bounds = [(val - 2.0, val + 2.0) for val in latent_vec] + [(-2.0, 2.0)]

        constraints = [
            {'type': 'ineq', 'fun': constraint},
            {'type': 'ineq', 'fun': min_latent_shift},
        ]

        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": constraints
        }

        result = basinhopping(objective, x0=latent_5d, minimizer_kwargs=minimizer_kwargs, niter=100)

        z_new = result.x
        z_new_latent = z_new[:4]
        z_new_tn = z_new[4]
        new_jump_duration = z_new_tn * jump_std + jump_mean
        predicted_new_height = predict_height(z_new)

        print("\n" + "=" * 60)
        print(f"Jump ID: {row['profileID']}, Date: {jump_date}")
        print(f"Predicted Original Height: {predicted_orig_height:.2f} inches")
        print(f"Target Prescribed Height: {target_jump_height:.2f} inches")
        print(f"Predicted Prescribed Height: {predicted_new_height:.2f} inches")
        print(f"Latent change magnitude: {np.linalg.norm(z_new - latent_5d):.4f}")
        print(f"Original duration: {jump_duration:.3f}s | New duration: {new_jump_duration:.3f}s")

        print(f"\nLatent Dimension Comparison:")
        for i, (orig, new) in enumerate(zip(latent_5d, z_new)):
            label = "t_n" if i == 4 else f"z{i+1}"
            print(f"{label:<5}{orig:>12.4f}{new:>12.4f}{(new - orig):>18.4f}")

        with torch.no_grad():
            rec_orig = vae.decode(torch.tensor(latent_vec, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()
            rec_new = vae.decode(torch.tensor(z_new_latent, dtype=torch.float32).unsqueeze(0)).squeeze().cpu().numpy()

        offset_orig = rec_orig[0] * (20 / initial_val)
        offset_new = rec_new[0] * (20 / initial_val)
        adj_orig = rec_orig - offset_orig
        adj_new = rec_new - offset_new

        def parse_bin_label(label):
            start, end = label.split('-')
            return float(start), float(end)

        unique_bins = model_df['jump_bin'].dropna().unique()
        bin_tuples = [(label, *parse_bin_label(label)) for label in unique_bins]
        bin_tuples.sort(key=lambda x: x[1])

        current_bin_label, next_bin_label = None, None
        for i, (label, start, end) in enumerate(bin_tuples):
            if start <= predicted_orig_height < end:
                current_bin_label = label
                if i + 1 < len(bin_tuples):
                    next_bin_label = bin_tuples[i + 1][0]
                break

        higher_bin_latents = model_df[model_df['JUMP_HEIGHT_INCHES'] > actual_recorded_height]
        if next_bin_label:
            higher_bin_latents = higher_bin_latents[higher_bin_latents['jump_bin'] == next_bin_label]

        if not higher_bin_latents.empty and 'downsampled_time_series' in higher_bin_latents.columns:
            downsampled_arrays = np.stack(higher_bin_latents['downsampled_time_series'].values)
            avg_downsampled_ts = np.mean(downsampled_arrays, axis=0)

            offset_higher_bin = avg_downsampled_ts[0] * (20 / initial_val)
            adj_higher_bin = avg_downsampled_ts - offset_higher_bin
            rmse_orig = np.sqrt(np.mean((adj_orig - adj_higher_bin) ** 2))
            rmse_new = np.sqrt(np.mean((adj_new - adj_higher_bin) ** 2))
            corr_orig, _ = pearsonr(adj_orig, adj_higher_bin)
            corr_new, _ = pearsonr(adj_new, adj_higher_bin)

            height_gain = predicted_new_height - predicted_orig_height

            num_points = len(adj_orig)
            time_ms = np.linspace(0, new_jump_duration * 1000, num_points)

            plt.figure(figsize=(12, 6))
            adj_orig_resampled = resample_waveform(adj_orig, jump_duration, new_jump_duration)
            adj_new_resampled = resample_waveform(adj_new, new_jump_duration, new_jump_duration)  

            time_ms_old = np.linspace(0, jump_duration * 1000, len(adj_orig_resampled))
            time_ms_new = np.linspace(0, new_jump_duration * 1000, len(adj_new_resampled))
            cutoff_idx_orig = np.argmax(adj_orig_resampled <= 0)
            if adj_orig_resampled[-1] > 0:
                cutoff_idx_orig = len(adj_orig_resampled)

            cutoff_idx_new = np.argmax(adj_new_resampled <= 0)
            if adj_new_resampled[-1] > 0:
                cutoff_idx_new = len(adj_new_resampled)

            cutoff_idx_new = max(cutoff_idx_orig, cutoff_idx_new)
            cutoff_idx_old = min(cutoff_idx_orig, cutoff_idx_new)


            time_ms_old = time_ms_old[:cutoff_idx_old]
            time_ms_new = time_ms_new[:cutoff_idx_new]
            adj_orig_resampled = adj_orig_resampled[:cutoff_idx_old]
            adj_new_resampled = adj_new_resampled[:cutoff_idx_new]

            plt.plot(time_ms_old, adj_orig_resampled, label=f'Original Jump ({predicted_orig_height:.1f}")', color='blue')
            plt.plot(time_ms_new, adj_new_resampled, label=f'Prescribed Jump ({predicted_new_height:.1f})')


            # plt.plot(time_ms, adj_orig, label=f'Original Jump ({predicted_orig_height:.1f}")', color='blue', linewidth=2)
            # plt.plot(time_ms, adj_new, label=f'Prescribed Jump ({predicted_new_height:.1f}")', color='green', linewidth=2)
            # if 'adj_higher_bin' in locals():
            #     plt.plot(time_ms, adj_higher_bin, label='Avg Higher Bin', linestyle='--', color='gray')

            # plt.fill_between(time_ms_old, adj_orig, adj_new, 
            #                  where=(adj_new > adj_orig), interpolate=True, 
            #                  color='lightgreen', alpha=0.3, label='Increased Force Region')
            # plt.fill_between(time_ms_new, adj_orig, adj_new, 
            #                  where=(adj_new < adj_orig), interpolate=True, 
            #                  color='lightcoral', alpha=0.3, label='Decreased Force Region')

            plt.title(f"Jump Shape Change ➜ Height Gain: {height_gain:.2f}\"", fontsize=14)
            plt.xlabel("Time (ms)")
            plt.ylabel("Adjusted Force")

            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            avg_force_diff = np.mean(adj_new - adj_orig)

            dt = (new_jump_duration / len(adj_orig))  # USE NEW jump duration here!
            impulse_diff = np.sum((adj_new - adj_orig) * dt)
            force_l2_diff = np.linalg.norm(adj_new - adj_orig)
            mass_kg = initial_val / 9.81  
            impulse_diff_newtons = impulse_diff * mass_kg * 9.81

        else:
            print("No higher bin data with downsampled_time_series available.")
