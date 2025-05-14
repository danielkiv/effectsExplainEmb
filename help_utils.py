
import torch
import numpy as np
import matplotlib.pyplot as plt
# --- Metric Calculation Imports ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import os, sys

# 1) Where is this script?
HERE = os.path.dirname(os.path.abspath(__file__))

# 2) Point at the TorchSpatial folder so that its .py files become top-level modules
TS_DIR = os.path.join(HERE, 'TorchSpatial')

# 3) Prepend it to sys.path
if TS_DIR not in sys.path:
    sys.path.insert(0, TS_DIR)

# 4) Now import exactly as TorchSpatial expects internally:
from SpatialRelationEncoder import *
from module import *
from data_utils import *
from utils import *

SPA_EMBED_DIM = 12  # Default embedding dimension for spatial encoders

def get_loc_embeddings(coords, encoder_type, device='cpu'):
    """
    Compute location embeddings for 2D coordinates using the specified spatial encoder type.

    Parameters:
        coords (np.ndarray): Array of shape [batch_size, 2] containing the coordinates.
        encoder_type (str): The string identifier for the spatial encoder (e.g., 'Space2Vec-grid', 'NeRF', etc).
        device (str): Device to use for the computation ('cpu', 'cuda:0', etc).

    Returns:
        torch.Tensor: The location embeddings, a tensor of shape [batch_size, spa_embed_dim].
    """
    # Define the parameter dictionary.
    params = {
        'spa_enc_type': encoder_type,   # use the provided encoder type
        'spa_embed_dim': SPA_EMBED_DIM,           # embedding dimension
        'extent': (0, 200, 0, 200),       # extent of the coordinates
        'freq': 16,                     # number of scales (related to multi-scale Fourier features)
        'max_radius': 1,                # maximum scale (lambda_max)
        'min_radius': 0.0001,           # minimum scale (lambda_min)
        'spa_f_act': 'leakyrelu',       # non-linear activation function
        'freq_init': 'geometric',       # Fourier frequency initialization
        'num_hidden_layer': 1,          # number of hidden layers in the encoder
        'dropout': 0.5,                 # dropout rate
        'hidden_dim': 512,              # hidden dimension of the MLP (if applicable)
        'use_layn': True,               # use layer normalization flag
        'skip_connection': True,        # apply skip connections
        'spa_enc_use_postmat': True,    # whether to use the post-processing matrix
        'device': device                # device for computation
    }

    # Instantiate the spatial relation encoder using the parameters.
    loc_enc = get_spa_encoder(
        train_locs=[],                      # no training coordinates provided here
        params=params,
        spa_enc_type=params['spa_enc_type'],
        spa_embed_dim=params['spa_embed_dim'],
        extent=params['extent'],
        coord_dim=2,                        # working in 2D
        frequency_num=params['freq'],
        max_radius=params['max_radius'],
        min_radius=params['min_radius'],
        f_act=params['spa_f_act'],
        freq_init=params['freq_init'],
        use_postmat=params['spa_enc_use_postmat'],
        device=params['device']
    ).to(params['device'])

    # Ensure coords is a NumPy array. If coords is 2D ([batch_size, 2]),
    # expand dims so that it has shape [batch_size, 1, 2] as required by the encoder.
    coords = np.array(coords)
    if coords.ndim == 2:
        coords = np.expand_dims(coords, axis=1)

    # Pass the coordinates through the encoder.
    # The mapping is mathematically represented as: loc_embeds = f(coords)
    loc_embeds = torch.squeeze(loc_enc(coords))
    return loc_embeds

# --- Plotting Function ---
# plot_s function remains the same as in the previous version
# (Handling list vmin/vmax, shape checks, saving logic)
def plot_s(bs, vmin=None, vmax=None, title="", filename=None, experiment_dir=None):
    """
    Plots spatial coefficient surfaces and saves the figure to a specific directory.
    Now handles potential vmin/vmax being passed for individual plots via lists.
    """
    if not isinstance(bs, list):
        if isinstance(bs, np.ndarray) and bs.ndim == 2 and bs.shape[1] == size * size:
             bs = [bs[i, :] for i in range(bs.shape[0])]
        elif isinstance(bs, np.ndarray) and bs.ndim == 1 and bs.shape[0] == size * size:
             bs = [bs]
        else:
             print(f"Error: Invalid input shape/type for plot_s: {type(bs)}. Expected list or array of shape ({size*size},).")
             if bs is not None: print(f"Actual shape: {bs.shape if isinstance(bs, np.ndarray) else 'N/A'}")
             return

    k = len(bs)
    fig, axs = plt.subplots(1, k, figsize=(6 * k, 4), dpi=300)
    if k == 1:
        axs = [axs]

    # Handle vmin/vmax potentially being lists
    vmin_list = vmin if isinstance(vmin, list) else [vmin] * k
    vmax_list = vmax if isinstance(vmax, list) else [vmax] * k
    if len(vmin_list) != k or len(vmax_list) != k:
        print("Warning: Length of vmin/vmax lists does not match number of plots. Using first value or None.")
        vmin_list = [vmin_list[0]] * k
        vmax_list = [vmax_list[0]] * k


    plots_successful = 0
    for i in range(k):
        current_vmin = vmin_list[i]
        current_vmax = vmax_list[i]
        if bs[i] is not None and hasattr(bs[i], 'shape') and bs[i].shape == (size * size,):
            # Check for constant array before plotting if vmin/vmax are identical
            is_constant = np.all(bs[i] == bs[i][0]) if bs[i].size > 0 else True
            if current_vmin is not None and current_vmax is not None and np.isclose(current_vmin, current_vmax):
                 if not is_constant:
                     print(f"Note: vmin ({current_vmin:.2f}) and vmax ({current_vmax:.2f}) are very close for non-constant data in plot {i}. Adjusting slightly.")
                     # Add a small buffer if they are identical but data isn't constant
                     buffer = 0.1 * abs(current_vmin) if abs(current_vmin) > 1e-6 else 0.1
                     current_vmin -= buffer
                     current_vmax += buffer
                 else:
                     # If data is constant, imshow might handle it, or we can skip colorbar
                     pass # Let imshow handle constant data with potentially identical vmin/vmax

            plot_data = bs[i].reshape(size, size)
            # Use try-except for imshow in case of rendering issues
            try:
                im = axs[i].imshow(plot_data, cmap='viridis', vmin=current_vmin, vmax=current_vmax)
                fig.colorbar(im, ax=axs[i])
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_xticklabels([])
                axs[i].set_yticklabels([])
                if isinstance(title, list) and i < len(title):
                     axs[i].set_title(title[i]) # Set title here
                plots_successful += 1
            except Exception as img_err:
                 print(f"Error during imshow/colorbar for plot {i}: {img_err}")
                 axs[i].text(0.5, 0.5, f'Plot Failed\n({type(img_err).__name__})', ha='center', va='center', transform=axs[i].transAxes, color='red')
                 plot_title = f"Component {i+1} (Failed)"
                 if isinstance(title, list) and i < len(title):
                     plot_title = f"{title[i]} (Failed)"
                 axs[i].set_title(plot_title)

        else:
            reason = 'Shape Mismatch' if bs[i] is not None and hasattr(bs[i], 'shape') else 'Data Missing/Invalid'
            actual_shape_info = f"Actual shape: {bs[i].shape}" if bs[i] is not None and hasattr(bs[i], 'shape') else ""
            print(f"Warning: Skipping plot for component {i} ({reason}). {actual_shape_info}")
            axs[i].text(0.5, 0.5, f'Plot Skipped\n({reason})', ha='center', va='center', transform=axs[i].transAxes, color='red')
            plot_title = f"Component {i+1} (Skipped)"
            if isinstance(title, list) and i < len(title):
                plot_title = f"{title[i]} (Skipped)"
            axs[i].set_title(plot_title)


    if isinstance(title, str) and title:
        # Use try-except for suptitle as well
        try:
            fig.suptitle(title, fontsize=16, y=1.02)
        except Exception as suptitle_err:
            print(f"Error setting suptitle '{title}': {suptitle_err}")


    # Use try-except for layout adjustments
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    except Exception as layout_err:
        print(f"Error during tight_layout: {layout_err}. Plot might overlap.")


    if filename and experiment_dir and plots_successful > 0:
        save_path = os.path.join(experiment_dir, filename)
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        except Exception as e:
            print(f"Error saving figure {save_path}: {e}")
        plt.close(fig)
    elif plots_successful == 0 and filename:
        print(f"No plots successful for '{filename}'. Closing figure.")
        plt.close(fig)
    elif not filename:
        plt.close(fig)


# --- Spatial Effect Metric Calculation Functions ---
# calculate_spatial_metrics function remains the same
def calculate_spatial_metrics(true_surface, estimated_surface, effect_name, encoder_name, model_name):
    """Calculates various metrics comparing true and estimated spatial surfaces."""
    if true_surface is None or estimated_surface is None:
        print(f"Warning: Skipping metrics for {effect_name} ({encoder_name}/{model_name}) due to missing data.")
        return None
    if true_surface.shape != estimated_surface.shape:
        print(f"Warning: Skipping metrics for {effect_name} ({encoder_name}/{model_name}) due to shape mismatch: True {true_surface.shape}, Est {estimated_surface.shape}")
        return None
    if true_surface.shape[0] == 0: # Handle empty arrays if they somehow occur
        print(f"Warning: Skipping metrics for {effect_name} ({encoder_name}/{model_name}) due to empty surface.")
        return None


    metrics = {
        'encoder': encoder_name,
        'model': model_name,
        'spatial_effect': effect_name,
    }
    try:
        # Filter NaNs before calculation if necessary
        mask = ~np.isnan(true_surface) & ~np.isnan(estimated_surface)
        if np.sum(mask) < 2: # Need at least 2 valid points for correlation/std dev
             print(f"Warning: Not enough valid data points ({np.sum(mask)}) for metrics calculation for {effect_name} ({encoder_name}/{model_name}).")
             # Return dict with NaNs instead of None to maintain structure
             metrics.update({m_key: np.nan for m_key in ['mse', 'rmse', 'mae', 'pearson_r', 'pearson_r_squared', 'r2_score', 'mean_error_bias']})
             return metrics

        true_valid = true_surface[mask]
        est_valid = estimated_surface[mask]

        metrics['mse'] = mean_squared_error(true_valid, est_valid)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(true_valid, est_valid)

        if np.std(true_valid) > 1e-6 and np.std(est_valid) > 1e-6:
            pearson_r, _ = pearsonr(true_valid, est_valid)
            metrics['pearson_r'] = pearson_r
            metrics['pearson_r_squared'] = pearson_r**2
        else:
            metrics['pearson_r'] = np.nan
            metrics['pearson_r_squared'] = np.nan
            # Don't print warning here if it's expected due to low variance data
            # print(f"Note: Pearson correlation not calculated for {effect_name} ({encoder_name}/{model_name}) due to constant true or estimated surface.")

        metrics['r2_score'] = r2_score(true_valid, est_valid)
        metrics['mean_error_bias'] = np.mean(est_valid - true_valid)
    except Exception as e:
        print(f"Error calculating metrics for {effect_name} ({encoder_name}/{model_name}): {e}")
        for m_key in ['mse', 'rmse', 'mae', 'pearson_r', 'pearson_r_squared', 'r2_score', 'mean_error_bias']:
            if m_key not in metrics: metrics[m_key] = np.nan
    return metrics