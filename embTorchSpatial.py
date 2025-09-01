
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
import torch # Added for potential tensor output from get_loc_embeddings

# --- Experiment Setup ---
BASE_EXPERIMENT_DIR = './results/dumb_multi_embedding_fixed' # Base directory for all results
print(f"Base experiment directory: {BASE_EXPERIMENT_DIR}")
os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)

# --- Warning Filtering ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
# warnings.filterwarnings('ignore', module='mgwr.*') # Uncomment if needed

# --- Imports ---
# Assuming geoshapley.py is in the same directory or PYTHONPATH
from geoshapley import GeoShapleyExplainer
# Assuming help_utils.py with get_loc_embeddings is in the same directory or PYTHONPATH
# If not, you might need to copy the function definition here or adjust the import path.
try:
    from help_utils import get_loc_embeddings
except ImportError:
    print("Error: Could not import get_loc_embeddings from help_utils.")
    print("Please ensure help_utils.py is accessible and contains the function.")
    # Define a placeholder function if import fails, to allow script structure to run
    # Replace this with the actual function if needed.
    def get_loc_embeddings(coords_array, encoder_type, device='cpu'):
        print(f"Placeholder: Generating dummy embeddings for {encoder_type}")
        # Return dummy embeddings matching expected output shape (N, D)
        # D=10 is arbitrary, adjust if needed or based on expected dim
        return np.random.rand(coords_array.shape[0], 10)
    # exit() # Optionally exit if the function is critical

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from mgwr.gwr import GWR, MGWR # Keep if MGWR analysis is still desired
from mgwr.sel_bw import Sel_BW # Keep if MGWR analysis is still desired

# --- Plotting Configuration ---
size = 25 # Grid size for spatial plots

# --- Plotting Function ---
def plot_s(bs, vmin=None, vmax=None, title="", filename=None, experiment_dir=None):
    """
    Plots spatial coefficient surfaces and saves the figure to a specific directory.

    Args:
        bs (list or np.ndarray): List/array of spatial coefficient surfaces (each size*size,).
        vmin (float, optional): Min value for colorbar.
        vmax (float, optional): Max value for colorbar.
        title (str or list, optional): Plot title(s).
        filename (str, optional): Filename (without path).
        experiment_dir (str, optional): Directory to save the plot in. Required if filename is provided.
    """
    # Ensure bs is a list of 1D arrays
    if not isinstance(bs, list):
        if isinstance(bs, np.ndarray) and bs.ndim == 2 and bs.shape[1] == size * size:
             bs = [bs[i, :] for i in range(bs.shape[0])] # Convert (k, N) to list of (N,)
        elif isinstance(bs, np.ndarray) and bs.ndim == 1 and bs.shape[0] == size * size:
             bs = [bs] # Single surface
        else:
             print(f"Error: Invalid input shape/type for plot_s: {type(bs)}. Expected list or array.")
             return

    k = len(bs)
    fig, axs = plt.subplots(1, k, figsize=(6 * k, 4), dpi=300)
    if k == 1:
        axs = [axs] # Make iterable

    plots_successful = 0
    for i in range(k):
        if bs[i] is not None and bs[i].shape[0] == size * size:
            plot_data = bs[i].reshape(size, size)
            im = axs[i].imshow(plot_data, cmap='viridis', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axs[i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
            if isinstance(title, list) and i < len(title):
                 axs[i].set_title(title[i])
            plots_successful += 1
        else:
            reason = 'Shape Mismatch' if bs[i] is not None else 'Data Missing'
            print(f"Warning: Skipping plot for component {i} ({reason}).")
            axs[i].text(0.5, 0.5, f'Plot Skipped\n({reason})', ha='center', va='center', transform=axs[i].transAxes, color='red')
            plot_title = f"Component {i+1} (Skipped)"
            if isinstance(title, list) and i < len(title):
                plot_title = f"{title[i]} (Skipped)"
            axs[i].set_title(plot_title)

    if isinstance(title, str) and title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if filename and experiment_dir and plots_successful > 0:
        save_path = os.path.join(experiment_dir, filename)
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        except Exception as e:
            print(f"Error saving figure {save_path}: {e}")
        plt.close(fig)
    elif plots_successful == 0:
        print(f"No plots successful for '{title}'. Closing figure.")
        plt.close(fig)
    # No plt.show() to avoid blocking in scripts; rely on saved files.


# --- Spatial Embedding Types ---
# Extracted from explainable_spatialeffects_embedding_test.py
encoder_types = [
    "Space2Vec-theory", "tile_ffn", "wrap_ffn",
    "Sphere2Vec-sphereM", "Sphere2Vec-sphereM+", "rff",
    "Sphere2Vec-sphereC", "Sphere2Vec-sphereC+", "NeRF",
    "Sphere2Vec-dfs", "Space2Vec-grid"
]
# encoder_types = ["rff", "NeRF"] # Subset for faster testing if needed


# --- Data Loading ---
print("Loading data...")
try:
    # Using a reliable URL for the data
    data_url = "https://raw.githubusercontent.com/Ziqi-Li/geoshapley/main/data/mgwr_sim.csv"
    mgwr_sim = pd.read_csv(data_url)
    print(f"Data loaded successfully from {data_url}.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Separate coordinates, features, target, and true coefficients
original_coords = mgwr_sim[['x_coord', 'y_coord']].copy()
X_features = mgwr_sim[['X1', 'X2']].copy() # Original non-spatial features
y = mgwr_sim.y.values
true_coeffs = mgwr_sim[["b0", "b1", "b2"]].values # True intercept and slopes

# Plot true coefficients (once, outside the loop)
print("Plotting true coefficients...")
if true_coeffs.shape[0] == size * size:
     plot_s(
         [true_coeffs[:, 0], true_coeffs[:, 1], true_coeffs[:, 2]], # List of 1D arrays
         vmin=1, vmax=5,
         title=["True Intercept", "True Coeff X1", "True Coeff X2"],
         filename="true_coefficients.pdf",
         experiment_dir=BASE_EXPERIMENT_DIR # Save in base directory
     )
else:
     print(f"Warning: True coefficients shape {true_coeffs.shape} != grid size {size*size}. Skipping plot.")


# --- Master Loop for Embeddings ---
all_model_metrics = {}

for encoder in encoder_types:
    print(f"\n===== Processing Encoder: {encoder} =====")

    # --- Create Encoder-Specific Directory ---
    CURRENT_EXPERIMENT_DIR = os.path.join(BASE_EXPERIMENT_DIR, encoder)
    print(f"Ensuring experiment directory exists: {CURRENT_EXPERIMENT_DIR}")
    os.makedirs(CURRENT_EXPERIMENT_DIR, exist_ok=True)

    # --- Generate Location Embeddings ---
    print(f"Generating {encoder} location features...")
    try:
        # Use the imported function
        # Pass coordinates as numpy array
        embeddings = get_loc_embeddings(original_coords.values, encoder_type=encoder, device='cpu')
        # Convert to numpy if it's a tensor
        embeddings_np = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)

        if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
            print(f"Warning: NaN or Inf values found in embeddings for {encoder}. Skipping this encoder.")
            all_model_metrics[encoder] = {'MLP': {'Train_R2': np.nan, 'Test_R2': np.nan},
                                          'XGBoost': {'Train_R2': np.nan, 'Test_R2': np.nan}}
            continue # Skip to the next encoder

        emb_dim = embeddings_np.shape[1]
        emb_cols = [f"{encoder}_emb_{i}" for i in range(emb_dim)]
        X_embeddings = pd.DataFrame(embeddings_np, columns=emb_cols, index=original_coords.index)
        print(f"Generated {emb_dim}-dimensional embeddings.")

    except Exception as e:
        print(f"Error generating embeddings for {encoder}: {e}")
        print(f"Skipping this encoder.")
        all_model_metrics[encoder] = {'MLP': {'Train_R2': np.nan, 'Test_R2': np.nan},
                                      'XGBoost': {'Train_R2': np.nan, 'Test_R2': np.nan}}
        continue # Skip to the next encoder


    # --- Prepare Data for Models (within the loop) ---
    # ML models use original features + current embeddings
    X_ml_features = pd.concat([X_features, X_embeddings], axis=1)
    ml_feature_names = X_ml_features.columns.tolist()
    print(f"ML model features ({X_ml_features.shape[1]}) for {encoder}: {ml_feature_names}")

    # GeoShapley uses ML features + original coordinates (at the end)
    X_for_geoshapley = pd.concat([X_ml_features, original_coords], axis=1)
    geoshapley_feature_names = X_for_geoshapley.columns.tolist()
    # print(f"GeoShapley input features ({X_for_geoshapley.shape[1]}): {geoshapley_feature_names}") # Can be verbose

    # Split data (using GeoShapley data structure for alignment)
    # Use the same random state for consistent splits across encoders
    print("\nSplitting data (80/20 split)...")
    X_train_gs, X_test_gs, y_train, y_test = train_test_split(
        X_for_geoshapley, y, test_size=0.20, random_state=42
    )

    # Separate ML features from the splits
    X_train_ml = X_train_gs[ml_feature_names]
    X_test_ml = X_test_gs[ml_feature_names]

    # Define GeoShapley background and explanation data
    background_data_gs = X_train_gs # Use the DataFrame directly
    explanation_data_gs = X_for_geoshapley # Use the DataFrame directly

    # print(f"ML Training data shape: {X_train_ml.shape}")
    # print(f"ML Test data shape: {X_test_ml.shape}")
    # print(f"GeoShapley Background data shape: {background_data_gs.shape}")
    # print(f"GeoShapley Explanation data shape: {explanation_data_gs.shape}")


    # --- Model Training (within the loop) ---
    model_metrics = {}

    print(f"\nTraining MLP model for {encoder}...")
    mlp_model = MLPRegressor(random_state=1, max_iter=600, hidden_layer_sizes=(64, 32), alpha=0.01, early_stopping=True)
    mlp_failed = False
    try:
        mlp_model.fit(X_train_ml, y_train)
        mlp_train_score = mlp_model.score(X_train_ml, y_train)
        mlp_test_score = mlp_model.score(X_test_ml, y_test)
        print(f"MLP Training successful. Train R^2: {mlp_train_score:.4f}, Test R^2: {mlp_test_score:.4f}")
        model_metrics['MLP'] = {'Train_R2': mlp_train_score, 'Test_R2': mlp_test_score}
    except Exception as e:
        print(f"MLP model training failed: {e}")
        mlp_failed = True
        model_metrics['MLP'] = {'Train_R2': np.nan, 'Test_R2': np.nan}


    print(f"\nTraining XGBoost model for {encoder}...")
    xgb_model = XGBRegressor(random_state=1, objective='reg:squarederror', n_estimators=100)
    xgb_failed = False
    try:
        xgb_model.fit(X_train_ml, y_train)
        xgb_train_score = xgb_model.score(X_train_ml, y_train)
        xgb_test_score = xgb_model.score(X_test_ml, y_test)
        print(f"XGBoost Training successful. Train R^2: {xgb_train_score:.4f}, Test R^2: {xgb_test_score:.4f}")
        model_metrics['XGBoost'] = {'Train_R2': xgb_train_score, 'Test_R2': xgb_test_score}
    except Exception as e:
        print(f"XGBoost model training failed: {e}")
        xgb_failed = True
        model_metrics['XGBoost'] = {'Train_R2': np.nan, 'Test_R2': np.nan}

    # Store metrics for this encoder
    all_model_metrics[encoder] = model_metrics

    # Save model metrics for this encoder
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "model_metrics.csv")
    try:
        metrics_df.to_csv(metrics_save_path)
        print(f"Saved model metrics to {metrics_save_path}")
    except Exception as e:
        print(f"Error saving model metrics: {e}")


    # --- GeoShapley Explanation (within the loop) ---

    # Indices of original features (X1, X2) within the GeoShapley data structure
    # These indices remain the same relative to the start of the dataframe
    original_feature_indices = [
        geoshapley_feature_names.index('X1'),
        geoshapley_feature_names.index('X2')
    ]
    # print(f"\nIndices of X1, X2 in GeoShapley data: {original_feature_indices}")

    # --- MLP Explanation ---
    print(f"\nExplaining MLP model with GeoShapley for {encoder}...")
    if not mlp_failed:
        try:
            # Prediction wrapper: Takes GeoShapley DF, selects ML features, predicts.
            def mlp_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     # Crucial: Ensure columns align with training features
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 # Select only the columns used for ML training *for this encoder*
                 X_ml = X_gs_df[ml_feature_names]
                 return mlp_model.predict(X_ml)

            # Initialize Explainer
            # Pass background data with features specific to this encoder
            mlp_explainer = GeoShapleyExplainer(mlp_predict_func, background_data_gs.values)

            # Explain using the entire dataset (with features specific to this encoder)
            mlp_rslt = mlp_explainer.explain(explanation_data_gs, n_jobs=-1)
            print("MLP explanation complete.")

            # Summary plot
            print("Generating MLP GeoShapley summary plot...")
            try:
                 mlp_rslt.summary_plot(dpi=150)
                 save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_plot.pdf")
                 plt.savefig(save_path, bbox_inches='tight')
                 print(f"Saved MLP summary plot: {save_path}")
                 plt.close('all')
            except Exception as e:
                print(f"Error saving MLP summary plot: {e}")
                plt.close('all')

            # Summary statistics
            print("Calculating MLP GeoShapley summary statistics...")
            try:
                # Pass the correct feature names including embeddings for this run
                summary_stats_mlp = mlp_rslt.summary_statistics()
                print("\nMLP GeoShapley Summary Statistics:")
                print(summary_stats_mlp)
                stats_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_statistics.csv")
                summary_stats_mlp.to_csv(stats_save_path)
                print(f"Saved MLP summary statistics: {stats_save_path}")
            except Exception as e:
                print(f"Error calculating/saving MLP summary statistics: {e}")


            # Plotting MLP SVCs
            print("\nPlotting MLP SVCs...")
            try:
                mlp_svc_interactions = mlp_rslt.get_svc(
                    col=original_feature_indices, coef_type="raw", include_primary=False
                )
                if mlp_svc_interactions is None or mlp_svc_interactions.shape[0] != size*size:
                     raise ValueError(f"MLP raw SVCs have unexpected shape or are None: {mlp_svc_interactions.shape if mlp_svc_interactions is not None else 'None'}")

                mlp_surfaces_to_plot = [
                    mlp_rslt.base_value + mlp_rslt.geo, # Approx Intercept
                    mlp_svc_interactions[:, 0],         # X1 Interaction SVC
                    mlp_svc_interactions[:, 1]          # X2 Interaction SVC
                ]
                plot_s(
                    mlp_surfaces_to_plot,
                    title=[f"MLP Intercept ({encoder})", f"MLP SVC X1 (Raw, {encoder})", f"MLP SVC X2 (Raw, {encoder})"],
                    filename="mlp_svc_raw.pdf",
                    experiment_dir=CURRENT_EXPERIMENT_DIR
                )

                # Plot Smoothed SVCs (Optional)
                print("Plotting MLP SVCs (Smoothed)...")
                mlp_svc_smoothed = mlp_rslt.get_svc(
                    col=original_feature_indices, coef_type="gwr", include_primary=False
                )
                if mlp_svc_smoothed is None or mlp_svc_smoothed.shape[0] != size*size:
                     raise ValueError(f"MLP smoothed SVCs have unexpected shape or are None: {mlp_svc_smoothed.shape if mlp_svc_smoothed is not None else 'None'}")

                mlp_smoothed_surfaces = [
                     mlp_rslt.base_value + mlp_rslt.geo,
                     mlp_svc_smoothed[:, 0],
                     mlp_svc_smoothed[:, 1]
                ]
                plot_s(
                    mlp_smoothed_surfaces,
                    title=[f"MLP Intercept ({encoder})", f"MLP SVC X1 (Smooth, {encoder})", f"MLP SVC X2 (Smooth, {encoder})"],
                    filename="mlp_svc_smoothed.pdf",
                    experiment_dir=CURRENT_EXPERIMENT_DIR
                )

            except Exception as e:
                 print(f"Error plotting MLP SVCs: {type(e).__name__}: {e}")

        except Exception as e:
            print(f"Error during MLP GeoShapley explanation process for {encoder}: {type(e).__name__}: {e}")
    else:
        print(f"Skipping MLP explanation for {encoder} due to training failure.")


    # --- XGBoost Explanation ---
    print(f"\nExplaining XGBoost model with GeoShapley for {encoder}...")
    if not xgb_failed:
        try:
            # Prediction wrapper for XGBoost
            def xgb_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 X_ml = X_gs_df[ml_feature_names] # Use correct features for this encoder
                 return xgb_model.predict(X_ml)

            # Initialize Explainer
            xgb_explainer = GeoShapleyExplainer(xgb_predict_func, background_data_gs.values)

            # Explain
            xgb_rslt = xgb_explainer.explain(explanation_data_gs, n_jobs=-1)
            print("XGBoost explanation complete.")

            # Summary plot
            print("Generating XGBoost GeoShapley summary plot...")
            try:
                xgb_rslt.summary_plot(dpi=150)
                save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "xgb_geoshapley_summary_plot.pdf")
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved XGBoost summary plot: {save_path}")
                plt.close('all')
            except Exception as e:
                 print(f"Error saving XGBoost summary plot: {e}")
                 plt.close('all')

            # Summary statistics
            print("Calculating XGBoost GeoShapley summary statistics...")
            try:
                # Pass correct feature names
                summary_stats_xgb = xgb_rslt.summary_statistics()
                print("\nXGBoost GeoShapley Summary Statistics:")
                print(summary_stats_xgb)
                stats_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "xgb_geoshapley_summary_statistics.csv")
                summary_stats_xgb.to_csv(stats_save_path)
                print(f"Saved XGBoost summary statistics: {stats_save_path}")
            except Exception as e:
                print(f"Error calculating/saving XGBoost summary statistics: {e}")


            # Plotting XGBoost SVCs
            print("\nPlotting XGBoost SVCs...")
            try:
                xgb_svc_interactions = xgb_rslt.get_svc(
                    col=original_feature_indices, coef_type="raw", include_primary=False
                )
                if xgb_svc_interactions is None or xgb_svc_interactions.shape[0] != size*size:
                    raise ValueError(f"XGBoost raw SVCs have unexpected shape or are None: {xgb_svc_interactions.shape if xgb_svc_interactions is not None else 'None'}")

                xgb_surfaces_to_plot = [
                    xgb_rslt.base_value + xgb_rslt.geo, # Approx Intercept
                    xgb_svc_interactions[:, 0],         # X1 Interaction SVC
                    xgb_svc_interactions[:, 1]          # X2 Interaction SVC
                ]
                plot_s(
                    xgb_surfaces_to_plot,
                    title=[f"XGB Intercept ({encoder})", f"XGB SVC X1 (Raw, {encoder})", f"XGB SVC X2 (Raw, {encoder})"],
                    filename="xgb_svc_raw.pdf",
                    experiment_dir=CURRENT_EXPERIMENT_DIR
                )

                # Plot Smoothed SVCs (Optional)
                print("Plotting XGBoost SVCs (Smoothed)...")
                xgb_svc_smoothed = xgb_rslt.get_svc(
                    col=original_feature_indices, coef_type="gwr", include_primary=False
                )
                if xgb_svc_smoothed is None or xgb_svc_smoothed.shape[0] != size*size:
                     raise ValueError(f"XGBoost smoothed SVCs have unexpected shape or are None: {xgb_svc_smoothed.shape if xgb_svc_smoothed is not None else 'None'}")

                xgb_smoothed_surfaces = [
                     xgb_rslt.base_value + xgb_rslt.geo,
                     xgb_svc_smoothed[:, 0],
                     xgb_svc_smoothed[:, 1]
                ]
                plot_s(
                    xgb_smoothed_surfaces,
                    title=[f"XGB Intercept ({encoder})", f"XGB SVC X1 (Smooth, {encoder})", f"XGB SVC X2 (Smooth, {encoder})"],
                    filename="xgb_svc_smoothed.pdf",
                    experiment_dir=CURRENT_EXPERIMENT_DIR
                )

            except Exception as e:
                print(f"Error plotting XGBoost SVCs: {type(e).__name__}: {e}")

        except Exception as e:
            print(f"Error during XGBoost GeoShapley explanation process for {encoder}: {type(e).__name__}: {e}")
    else:
        print(f"Skipping XGBoost explanation for {encoder} due to training failure.")

    print(f"===== Finished Encoder: {encoder} =====")


# --- Save Combined Model Metrics ---
print("\nSaving combined model metrics across all encoders...")
# Combine metrics from the dictionary
combined_metrics_list = []
for encoder, metrics in all_model_metrics.items():
    for model, scores in metrics.items():
        combined_metrics_list.append({
            'encoder': encoder,
            'model': model,
            'Train_R2': scores.get('Train_R2', np.nan),
            'Test_R2': scores.get('Test_R2', np.nan)
        })

combined_metrics_df = pd.DataFrame(combined_metrics_list)
combined_metrics_save_path = os.path.join(BASE_EXPERIMENT_DIR, "all_encoder_model_metrics.csv")
try:
    combined_metrics_df.to_csv(combined_metrics_save_path, index=False)
    print(f"Saved combined model metrics to {combined_metrics_save_path}")
    print("\nCombined Metrics Summary:")
    print(combined_metrics_df)
except Exception as e:
    print(f"Error saving combined model metrics: {e}")


# --- MGWR Analysis (Optional - Outside the loop, uses original features) ---
run_mgwr = False # Set to True to run MGWR analysis
if run_mgwr:
    print("\nFitting MGWR model (using original coordinates and X1, X2)...")
    coords_mgwr = original_coords.values
    X_mgwr = X_features.values # Original features only
    y_mgwr = y.reshape(-1, 1)

    try:
        print("Selecting MGWR bandwidths...")
        sel = Sel_BW(coords_mgwr, y_mgwr, X_mgwr, multi=True, spherical=False)
        bw = sel.search(bw_min=2, multi_bw_min=[2] * X_mgwr.shape[1], verbose=False)
        print(f"MGWR bandwidths selected: {bw}")

        mgwr_model = MGWR(coords_mgwr, y_mgwr, X_mgwr, selector=sel, fixed=False, spherical=False)
        mgwr_results = mgwr_model.fit()
        print("MGWR fitting complete.")
        print(mgwr_results.summary())

        # Plot MGWR estimates
        print("Plotting MGWR estimates...")
        if mgwr_results.params.shape[1] == X_mgwr.shape[1] + 1: # Intercept + slopes
            mgwr_coeffs_to_plot = [mgwr_results.params[:, i] for i in range(mgwr_results.params.shape[1])]
            mgwr_titles = ["MGWR Intercept"] + [f"MGWR Coeff {col}" for col in X_features.columns]
            plot_s(
                mgwr_coeffs_to_plot,
                vmin=1, vmax=5,
                title=mgwr_titles,
                filename="mgwr_coefficient_estimates.pdf",
                experiment_dir=BASE_EXPERIMENT_DIR # Save in base directory
            )
        else:
             print(f"Warning: MGWR params shape {mgwr_results.params.shape} unexpected. Skipping plot.")

        # Save MGWR summary text
        mgwr_summary_path = os.path.join(BASE_EXPERIMENT_DIR, "mgwr_summary.txt")
        try:
            with open(mgwr_summary_path, "w") as f:
                f.write(str(mgwr_results.summary()))
            print(f"Saved MGWR summary: {mgwr_summary_path}")
        except Exception as e:
            print(f"Error saving MGWR summary: {e}")

    except Exception as e:
        print(f"Error during MGWR analysis: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nSkipping MGWR analysis as run_mgwr is False.")


print("\nScript finished.")