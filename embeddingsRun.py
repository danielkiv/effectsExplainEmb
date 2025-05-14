# Previous code... (Imports, Setup, Functions remain the same)
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
import torch # Added for potential tensor output from get_loc_embeddings

# --- Experiment Setup ---
BASE_EXPERIMENT_DIR = './results/dumb_multi_embedding_fixed2' # Base directory for all results
print(f"Base experiment directory: {BASE_EXPERIMENT_DIR}")
os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)

# --- Warning Filtering ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
# warnings.filterwarnings('ignore', module='mgwr.*') # Uncomment if needed

from geoshapley import GeoShapleyExplainer
from help_utils import get_loc_embeddings, plot_s, calculate_spatial_metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

# --- Plotting Configuration ---
SIZE = 25 # Grid size for spatial plots
# Define percentiles for clipping raw SVC color scales
PERCENTILE_LOWER = 1.0 # e.g., 1st percentile
PERCENTILE_UPPER = 99.0 # e.g., 99th percentile

# --- Spatial Embedding Types ---
encoder_types = [
    "Space2Vec-theory", "tile_ffn", "wrap_ffn",
    "Sphere2Vec-sphereM", "Sphere2Vec-sphereM+", "rff",
    "Sphere2Vec-sphereC", "Sphere2Vec-sphereC+", "NeRF",
    "Sphere2Vec-dfs", "Space2Vec-grid"
]
# encoder_types = ["NeRF"] # Subset for faster testing

# --- Data Loading ---
print("Loading data...")
try:
    data_dir = "./data/mgwr_sim.csv" # Adjust as needed
    mgwr_sim = pd.read_csv(data_dir)
    print(f"Data loaded successfully from {data_dir}.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

original_coords = mgwr_sim[['x_coord', 'y_coord']].copy()
X_features = mgwr_sim[['X1', 'X2']].copy()
y = mgwr_sim.y.values
true_coeffs_df = mgwr_sim[["b0", "b1", "b2"]].copy() # True intercept and slopes

true_intercept_surface = true_coeffs_df["b0"].values.flatten()
true_svc_x1_surface = true_coeffs_df["b1"].values.flatten()
true_svc_x2_surface = true_coeffs_df["b2"].values.flatten()

if not (true_intercept_surface.shape == (SIZE*SIZE,) and \
        true_svc_x1_surface.shape == (SIZE*SIZE,) and \
        true_svc_x2_surface.shape == (SIZE*SIZE,)):
    print(f"FATAL ERROR: True coefficient surfaces do not match expected shape ({SIZE*SIZE},).")
    exit()

print("Plotting true coefficients...")
# --- FIX: Use plain text for true coefficient titles ---
plot_s(
    [true_intercept_surface, true_svc_x1_surface, true_svc_x2_surface],
    size=SIZE,
    vmin=1, vmax=5,
    title=["True Intercept (b0)", "True Coeff X1 (b1)", "True Coeff X2 (b2)"], # Plain text titles
    filename="true_coefficients.pdf",
    experiment_dir=BASE_EXPERIMENT_DIR
)

# --- Master Loop for Embeddings ---
all_model_metrics = {}
all_spatial_effect_capture_metrics = []

for encoder in encoder_types:
    print(f"\n===== Processing Encoder: {encoder} =====")
    CURRENT_EXPERIMENT_DIR = os.path.join(BASE_EXPERIMENT_DIR, encoder)
    print(f"Ensuring experiment directory exists: {CURRENT_EXPERIMENT_DIR}")
    os.makedirs(CURRENT_EXPERIMENT_DIR, exist_ok=True)

    print(f"Generating {encoder} location features...")
    try:
        embeddings = get_loc_embeddings(original_coords.values, encoder_type=encoder, device='cpu')
        embeddings_np = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
        if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
            print(f"Warning: NaN or Inf values found in embeddings for {encoder}. Skipping this encoder.")
            all_model_metrics[encoder] = {'MLP': {'Train_R2': np.nan, 'Test_R2': np.nan}, 'XGBoost': {'Train_R2': np.nan, 'Test_R2': np.nan}}
            continue
        emb_dim = embeddings_np.shape[1]
        emb_cols = [f"{encoder}_emb_{i}" for i in range(emb_dim)]
        X_embeddings = pd.DataFrame(embeddings_np, columns=emb_cols, index=original_coords.index)
        print(f"Generated {emb_dim}-dimensional embeddings.")
    except Exception as e:
        print(f"Error generating embeddings for {encoder}: {e}")
        all_model_metrics[encoder] = {'MLP': {'Train_R2': np.nan, 'Test_R2': np.nan}, 'XGBoost': {'Train_R2': np.nan, 'Test_R2': np.nan}}
        continue

    X_ml_features = pd.concat([X_features, X_embeddings], axis=1)
    ml_feature_names = X_ml_features.columns.tolist()
    X_for_geoshapley = pd.concat([X_ml_features, original_coords], axis=1)
    geoshapley_feature_names = X_for_geoshapley.columns.tolist()

    print("\nSplitting data (80/20 split)...")
    X_train_gs, X_test_gs, y_train, y_test = train_test_split(
        X_for_geoshapley, y, test_size=0.20, random_state=42
    )
    X_train_ml = X_train_gs[ml_feature_names]
    X_test_ml = X_test_gs[ml_feature_names]
    background_data_gs = X_train_gs
    explanation_data_gs = X_for_geoshapley

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

    all_model_metrics[encoder] = model_metrics
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
    metrics_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "model_performance_metrics.csv")
    try:
        metrics_df.to_csv(metrics_save_path)
        print(f"Saved model performance metrics to {metrics_save_path}")
    except Exception as e:
        print(f"Error saving model performance metrics: {e}")

    original_feature_indices = [
        geoshapley_feature_names.index('X1'),
        geoshapley_feature_names.index('X2')
    ]

    # --- MLP Explanation and Spatial Metrics ---
    print(f"\nExplaining MLP model with GeoShapley for {encoder}...")
    current_model_spatial_metrics = []
    if not mlp_failed:
        try:
            def mlp_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 X_ml = X_gs_df[ml_feature_names]
                 return mlp_model.predict(X_ml)
            mlp_explainer = GeoShapleyExplainer(mlp_predict_func, background_data_gs.values)
            mlp_rslt = mlp_explainer.explain(explanation_data_gs, n_jobs=-1)
            print("MLP explanation complete.")

            try:
                 mlp_rslt.summary_plot(dpi=150, feature_names=geoshapley_feature_names[:-2])
                 save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_plot.pdf")
                 plt.savefig(save_path, bbox_inches='tight'); plt.close('all')
                 print(f"Saved MLP summary plot: {save_path}")
                 summary_stats_mlp = mlp_rslt.summary_statistics(feature_names=geoshapley_feature_names[:-2])
                 stats_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_statistics.csv")
                 summary_stats_mlp.to_csv(stats_save_path); print(f"Saved MLP summary statistics: {stats_save_path}")
            except Exception as e: print(f"Error in MLP summary plot/stats: {e}"); plt.close('all')

            est_intercept_mlp = mlp_rslt.base_value + mlp_rslt.geo
            if est_intercept_mlp.shape == true_intercept_surface.shape:
                metrics = calculate_spatial_metrics(true_intercept_surface, est_intercept_mlp, "Intercept", encoder, "MLP")
                if metrics: current_model_spatial_metrics.append(metrics)

            mlp_svc_interactions_raw = mlp_rslt.get_svc(col=original_feature_indices, coef_type="raw", include_primary=False)
            if mlp_svc_interactions_raw is not None and mlp_svc_interactions_raw.shape == (SIZE*SIZE, len(original_feature_indices)):
                vmin_x1_raw, vmax_x1_raw = np.nanpercentile(mlp_svc_interactions_raw[:, 0], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                vmin_x2_raw, vmax_x2_raw = np.nanpercentile(mlp_svc_interactions_raw[:, 1], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                print(f"MLP Raw SVC X1 clipped scale ({PERCENTILE_LOWER}-{PERCENTILE_UPPER}%): {vmin_x1_raw:.2f} to {vmax_x1_raw:.2f}")
                print(f"MLP Raw SVC X2 clipped scale ({PERCENTILE_LOWER}-{PERCENTILE_UPPER}%): {vmin_x2_raw:.2f} to {vmax_x2_raw:.2f}")

                if mlp_svc_interactions_raw[:, 0].shape == true_svc_x1_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, mlp_svc_interactions_raw[:, 0], "SVC_X1_Raw", encoder, "MLP")
                    if metrics: current_model_spatial_metrics.append(metrics)
                if mlp_svc_interactions_raw[:, 1].shape == true_svc_x2_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, mlp_svc_interactions_raw[:, 1], "SVC_X2_Raw", encoder, "MLP")
                    if metrics: current_model_spatial_metrics.append(metrics)

                # --- FIX: Use plain text titles ---
                plot_s([est_intercept_mlp, mlp_svc_interactions_raw[:, 0], mlp_svc_interactions_raw[:, 1]],
                       size=SIZE,
                       vmin=[None, vmin_x1_raw, vmin_x2_raw],
                       vmax=[None, vmax_x1_raw, vmax_x2_raw],
                       title=[f"MLP Intercept ({encoder})", f"MLP SVC X1 (Raw Clp {PERCENTILE_LOWER}-{PERCENTILE_UPPER}%, {encoder})", f"MLP SVC X2 (Raw Clp {PERCENTILE_LOWER}-{PERCENTILE_UPPER}%, {encoder})"],
                       filename="mlp_svc_raw_clipped.pdf",
                       experiment_dir=CURRENT_EXPERIMENT_DIR)

            mlp_svc_interactions_smooth = mlp_rslt.get_svc(col=original_feature_indices, coef_type="gwr", include_primary=False)
            if mlp_svc_interactions_smooth is not None and mlp_svc_interactions_smooth.shape == (SIZE*SIZE, len(original_feature_indices)):
                if mlp_svc_interactions_smooth[:, 0].shape == true_svc_x1_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, mlp_svc_interactions_smooth[:, 0], "SVC_X1_Smooth", encoder, "MLP")
                    if metrics: current_model_spatial_metrics.append(metrics)
                if mlp_svc_interactions_smooth[:, 1].shape == true_svc_x2_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, mlp_svc_interactions_smooth[:, 1], "SVC_X2_Smooth", encoder, "MLP")
                    if metrics: current_model_spatial_metrics.append(metrics)

                # --- FIX: Use plain text titles ---
                plot_s([est_intercept_mlp, mlp_svc_interactions_smooth[:, 0], mlp_svc_interactions_smooth[:, 1]],
                       size=SIZE,
                       title=[f"MLP Intercept ({encoder})", f"MLP SVC X1 (Smooth, {encoder})", f"MLP SVC X2 (Smooth, {encoder})"],
                       filename="mlp_svc_smoothed.pdf", experiment_dir=CURRENT_EXPERIMENT_DIR)
        except Exception as e: print(f"Error during MLP GeoShapley explanation or metrics for {encoder}: {type(e).__name__}: {e}")
    else: print(f"Skipping MLP explanation for {encoder} due to training failure.")

    # --- XGBoost Explanation and Spatial Metrics ---
    print(f"\nExplaining XGBoost model with GeoShapley for {encoder}...")
    if not xgb_failed:
        try:
            def xgb_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 X_ml = X_gs_df[ml_feature_names]
                 return xgb_model.predict(X_ml)
            xgb_explainer = GeoShapleyExplainer(xgb_predict_func, background_data_gs.values)
            xgb_rslt = xgb_explainer.explain(explanation_data_gs, n_jobs=-1)
            print("XGBoost explanation complete.")

            try:
                xgb_rslt.summary_plot(dpi=150, feature_names=geoshapley_feature_names[:-2])
                save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "xgb_geoshapley_summary_plot.pdf")
                plt.savefig(save_path, bbox_inches='tight'); plt.close('all')
                print(f"Saved XGBoost summary plot: {save_path}")
                summary_stats_xgb = xgb_rslt.summary_statistics(feature_names=geoshapley_feature_names[:-2])
                stats_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "xgb_geoshapley_summary_statistics.csv")
                summary_stats_xgb.to_csv(stats_save_path); print(f"Saved XGBoost summary statistics: {stats_save_path}")
            except Exception as e: print(f"Error in XGBoost summary plot/stats: {e}"); plt.close('all')

            est_intercept_xgb = xgb_rslt.base_value + xgb_rslt.geo
            if est_intercept_xgb.shape == true_intercept_surface.shape:
                metrics = calculate_spatial_metrics(true_intercept_surface, est_intercept_xgb, "Intercept", encoder, "XGBoost")
                if metrics: current_model_spatial_metrics.append(metrics)

            xgb_svc_interactions_raw = xgb_rslt.get_svc(col=original_feature_indices, coef_type="raw", include_primary=False)
            if xgb_svc_interactions_raw is not None and xgb_svc_interactions_raw.shape == (SIZE*SIZE, len(original_feature_indices)):
                vmin_x1_raw_xgb, vmax_x1_raw_xgb = np.nanpercentile(xgb_svc_interactions_raw[:, 0], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                vmin_x2_raw_xgb, vmax_x2_raw_xgb = np.nanpercentile(xgb_svc_interactions_raw[:, 1], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                print(f"XGB Raw SVC X1 clipped scale ({PERCENTILE_LOWER}-{PERCENTILE_UPPER}%): {vmin_x1_raw_xgb:.2f} to {vmax_x1_raw_xgb:.2f}")
                print(f"XGB Raw SVC X2 clipped scale ({PERCENTILE_LOWER}-{PERCENTILE_UPPER}%): {vmin_x2_raw_xgb:.2f} to {vmax_x2_raw_xgb:.2f}")

                if xgb_svc_interactions_raw[:, 0].shape == true_svc_x1_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, xgb_svc_interactions_raw[:, 0], "SVC_X1_Raw", encoder, "XGBoost")
                    if metrics: current_model_spatial_metrics.append(metrics)
                if xgb_svc_interactions_raw[:, 1].shape == true_svc_x2_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, xgb_svc_interactions_raw[:, 1], "SVC_X2_Raw", encoder, "XGBoost")
                    if metrics: current_model_spatial_metrics.append(metrics)

                # --- FIX: Use plain text titles ---
                plot_s([est_intercept_xgb, xgb_svc_interactions_raw[:, 0], xgb_svc_interactions_raw[:, 1]],
                       size=SIZE,
                       vmin=[None, vmin_x1_raw_xgb, vmin_x2_raw_xgb],
                       vmax=[None, vmax_x1_raw_xgb, vmax_x2_raw_xgb],
                       title=[f"XGB Intercept ({encoder})", f"XGB SVC X1 (Raw Clp {PERCENTILE_LOWER}-{PERCENTILE_UPPER}%, {encoder})", f"XGB SVC X2 (Raw Clp {PERCENTILE_LOWER}-{PERCENTILE_UPPER}%, {encoder})"],
                       filename="xgb_svc_raw_clipped.pdf",
                       experiment_dir=CURRENT_EXPERIMENT_DIR)

            xgb_svc_interactions_smooth = xgb_rslt.get_svc(col=original_feature_indices, coef_type="gwr", include_primary=False)
            if xgb_svc_interactions_smooth is not None and xgb_svc_interactions_smooth.shape == (SIZE*SIZE, len(original_feature_indices)):
                if xgb_svc_interactions_smooth[:, 0].shape == true_svc_x1_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, xgb_svc_interactions_smooth[:, 0], "SVC_X1_Smooth", encoder, "XGBoost")
                    if metrics: current_model_spatial_metrics.append(metrics)
                if xgb_svc_interactions_smooth[:, 1].shape == true_svc_x2_surface.shape:
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, xgb_svc_interactions_smooth[:, 1], "SVC_X2_Smooth", encoder, "XGBoost")
                    if metrics: current_model_spatial_metrics.append(metrics)

                # --- FIX: Use plain text titles ---
                plot_s([est_intercept_xgb, xgb_svc_interactions_smooth[:, 0], xgb_svc_interactions_smooth[:, 1]],
                       size=SIZE,
                       title=[f"XGB Intercept ({encoder})", f"XGB SVC X1 (Smooth, {encoder})", f"XGB SVC X2 (Smooth, {encoder})"],
                       filename="xgb_svc_smoothed.pdf", experiment_dir=CURRENT_EXPERIMENT_DIR)
        except Exception as e: print(f"Error during XGBoost GeoShapley explanation or metrics for {encoder}: {type(e).__name__}: {e}")
    else: print(f"Skipping XGBoost explanation for {encoder} due to training failure.")

    # Save spatial effect metrics for this encoder and its models
    if current_model_spatial_metrics:
        spatial_metrics_df = pd.DataFrame(current_model_spatial_metrics)
        spatial_metrics_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "spatial_effect_capture_metrics.csv")
        try:
            spatial_metrics_df.to_csv(spatial_metrics_save_path, index=False)
            print(f"Saved spatial effect capture metrics to {spatial_metrics_save_path}")
            all_spatial_effect_capture_metrics.extend(current_model_spatial_metrics) # Add to master list
        except Exception as e:
            print(f"Error saving spatial effect capture metrics: {e}")

    print(f"===== Finished Encoder: {encoder} =====")

# --- Save Combined Model Performance Metrics ---
# (Remains the same)
print("\nSaving combined model performance metrics across all encoders...")
combined_metrics_list = []
for encoder_name, metrics_dict in all_model_metrics.items():
    for model_name, scores_dict in metrics_dict.items():
        combined_metrics_list.append({
            'encoder': encoder_name,
            'model': model_name,
            'Train_R2': scores_dict.get('Train_R2', np.nan),
            'Test_R2': scores_dict.get('Test_R2', np.nan)
        })
combined_metrics_df = pd.DataFrame(combined_metrics_list)
combined_metrics_save_path = os.path.join(BASE_EXPERIMENT_DIR, "all_encoder_model_performance_metrics.csv")
try:
    combined_metrics_df.to_csv(combined_metrics_save_path, index=False)
    print(f"Saved combined model performance metrics to {combined_metrics_save_path}")
    print("\nCombined Model Performance Metrics Summary:")
    print(combined_metrics_df)
except Exception as e:
    print(f"Error saving combined model performance metrics: {e}")


# --- Save Combined Spatial Effect Capture Metrics ---
# (Remains the same)
if all_spatial_effect_capture_metrics:
    print("\nSaving combined spatial effect capture metrics across all experiments...")
    combined_spatial_metrics_df = pd.DataFrame(all_spatial_effect_capture_metrics)
    combined_spatial_metrics_save_path = os.path.join(BASE_EXPERIMENT_DIR, "all_experiments_spatial_effect_capture_metrics.csv")
    try:
        combined_spatial_metrics_df.to_csv(combined_spatial_metrics_save_path, index=False)
        print(f"Saved combined spatial effect capture metrics to {combined_spatial_metrics_save_path}")
        print("\nCombined Spatial Effect Capture Metrics Summary (First 5 rows):")
        print(combined_spatial_metrics_df.head())
    except Exception as e:
        print(f"Error saving combined spatial effect capture metrics: {e}")
else:
    print("\nNo spatial effect capture metrics were collected to save globally.")


# --- MGWR Analysis (Optional) ---
# (Remains the same, ensure titles are plain text if run)
run_mgwr = False
if run_mgwr:
    print("\nFitting MGWR model (using original coordinates and X1, X2)...")
    coords_mgwr = original_coords.values
    X_mgwr = X_features.values
    y_mgwr = y.reshape(-1, 1)
    try:
        print("Selecting MGWR bandwidths...")
        sel = Sel_BW(coords_mgwr, y_mgwr, X_mgwr, multi=True, spherical=False)
        bw = sel.search(bw_min=2, multi_bw_min=[2] * X_mgwr.shape[1], verbose=False) # Adjusted multi_bw_min
        print(f"MGWR bandwidths selected: {bw}")
        mgwr_model = MGWR(coords_mgwr, y_mgwr, X_mgwr, selector=sel, fixed=False, spherical=False) # Pass selector object
        mgwr_results = mgwr_model.fit()
        print("MGWR fitting complete.")
        print(mgwr_results.summary())
        # Plotting and saving MGWR results (existing code)
        if mgwr_results.params.shape[1] == X_mgwr.shape[1] + 1:
            mgwr_coeffs_to_plot = [mgwr_results.params[:, i] for i in range(mgwr_results.params.shape[1])]
            # --- FIX: Use plain text titles ---
            mgwr_titles = ["MGWR Intercept"] + [f"MGWR Coeff {col}" for col in X_features.columns]
            plot_s(mgwr_coeffs_to_plot, size=SIZE, vmin=1, vmax=5, title=mgwr_titles,
                   filename="mgwr_coefficient_estimates.pdf", experiment_dir=BASE_EXPERIMENT_DIR)
        else: print(f"Warning: MGWR params shape {mgwr_results.params.shape} unexpected. Skipping plot.")
        mgwr_summary_path = os.path.join(BASE_EXPERIMENT_DIR, "mgwr_summary.txt")
        with open(mgwr_summary_path, "w") as f: f.write(str(mgwr_results.summary()))
        print(f"Saved MGWR summary: {mgwr_summary_path}")
    except Exception as e: print(f"Error during MGWR analysis: {type(e).__name__}: {e}"); import traceback; traceback.print_exc()
else:
    print("\nSkipping MGWR analysis as run_mgwr is False.")

print("\nScript finished.")
