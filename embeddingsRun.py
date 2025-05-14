# Previous code... (Imports, Setup, Functions remain the same)
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
import torch # Added for potential tensor output from get_loc_embeddings
from libpysal import weights
from esda.moran import Moran

from geoshapley import GeoShapleyExplainer
from help_utils import get_loc_embeddings, plot_s, calculate_spatial_metrics # Ensure calculate_spatial_metrics is imported
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# from mgwr.gwr import GWR, MGWR # MGWR parts are optional
# from mgwr.sel_bw import Sel_BW # MGWR parts are optional

# --- Experiment Setup ---
BASE_EXPERIMENT_DIR = './results/dumb_multi_embedding_fixed2' # Base directory for all results
print(f"Base experiment directory: {BASE_EXPERIMENT_DIR}")
os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)

# --- Plotting Configuration ---
SIZE = 25 # Grid size for spatial plots
PERCENTILE_LOWER = 1.0
PERCENTILE_UPPER = 99.0

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
    data_dir = "./data/mgwr_sim.csv"
    mgwr_sim = pd.read_csv(data_dir)
    print(f"Data loaded successfully from {data_dir}.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

original_coords = mgwr_sim[['x_coord', 'y_coord']].copy() # This will be passed for Moran's I
X_features = mgwr_sim[['X1', 'X2']].copy()
y = mgwr_sim.y.values
true_coeffs_df = mgwr_sim[["b0", "b1", "b2"]].copy()

true_intercept_surface = true_coeffs_df["b0"].values.flatten()
true_svc_x1_surface = true_coeffs_df["b1"].values.flatten()
true_svc_x2_surface = true_coeffs_df["b2"].values.flatten()

if not (true_intercept_surface.shape == (SIZE*SIZE,) and \
        true_svc_x1_surface.shape == (SIZE*SIZE,) and \
        true_svc_x2_surface.shape == (SIZE*SIZE,)):
    print(f"FATAL ERROR: True coefficient surfaces do not match expected shape ({SIZE*SIZE},).")
    exit()

print("Plotting true coefficients...")
plot_s(
    [true_intercept_surface, true_svc_x1_surface, true_svc_x2_surface],
    size=SIZE,
    vmin=1, vmax=5,
    title=["True Intercept (b0)", "True Coeff X1 (b1)", "True Coeff X2 (b2)"],
    filename="true_coefficients.pdf",
    experiment_dir=BASE_EXPERIMENT_DIR
)

# --- Master Loop for Embeddings ---
all_model_metrics = {}
all_spatial_effect_capture_metrics = []

# ++ Crucial: Get the coordinates that correspond to the full surfaces for Moran's I calculation ++
# These coordinates should align with true_intercept_surface, etc.
# Assuming mgwr_sim is sorted such that its coordinates directly map to the flattened surfaces
coords_for_moran_full = original_coords.values # Use .values to get numpy array

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
    X_for_geoshapley = pd.concat([X_ml_features, original_coords], axis=1) # original_coords already here
    geoshapley_feature_names = X_for_geoshapley.columns.tolist()

    print("\nSplitting data (80/20 split)...")
    # Note: The coordinates for GeoShapley explanation (explanation_data_gs[['x_coord', 'y_coord']])
    # will be used for Moran's I on the *explained* surfaces.
    # These are the same as coords_for_moran_full if explanation_data_gs covers all points.
    X_train_gs, X_test_gs, y_train, y_test = train_test_split(
        X_for_geoshapley, y, test_size=0.20, random_state=42
    )
    X_train_ml = X_train_gs[ml_feature_names]
    X_test_ml = X_test_gs[ml_feature_names]
    background_data_gs = X_train_gs
    explanation_data_gs = X_for_geoshapley # Explaining on the full dataset
    
    # ++ Extract coordinates from explanation_data_gs for Moran's I related to GeoShapley outputs ++
    coords_for_explained_surfaces = explanation_data_gs[['x_coord', 'y_coord']].values

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

    current_model_spatial_metrics = []
    if not mlp_failed:
        print(f"\nExplaining MLP model with GeoShapley for {encoder}...")
        try:
            def mlp_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 X_ml = X_gs_df[ml_feature_names]
                 return mlp_model.predict(X_ml)
            mlp_explainer = GeoShapleyExplainer(mlp_predict_func, background_data_gs.values)
            mlp_rslt = mlp_explainer.explain(explanation_data_gs.values, n_jobs=-1) # Pass .values here
            print("MLP explanation complete.")

            # ... (summary plot and stats saving) ...
            try:
                 mlp_rslt.summary_plot(dpi=150, feature_names=geoshapley_feature_names[:-2]) # Pass original feature names without coords
                 save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_plot.pdf")
                 plt.savefig(save_path, bbox_inches='tight'); plt.close('all')
                 print(f"Saved MLP summary plot: {save_path}")
                 summary_stats_mlp = mlp_rslt.summary_statistics(feature_names=geoshapley_feature_names[:-2])
                 stats_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "mlp_geoshapley_summary_statistics.csv")
                 summary_stats_mlp.to_csv(stats_save_path); print(f"Saved MLP summary statistics: {stats_save_path}")
            except Exception as e: print(f"Error in MLP summary plot/stats: {e}"); plt.close('all')


            est_intercept_mlp = mlp_rslt.base_value + mlp_rslt.geo
            if est_intercept_mlp.shape == true_intercept_surface.shape:
                # ++ Pass coords_for_explained_surfaces ++
                metrics = calculate_spatial_metrics(true_intercept_surface, est_intercept_mlp, "Intercept", encoder, "MLP", coords_for_explained_surfaces)
                if metrics: current_model_spatial_metrics.append(metrics)

            mlp_svc_interactions_raw = mlp_rslt.get_svc(col=original_feature_indices, coef_type="raw", include_primary=False)
            if mlp_svc_interactions_raw is not None and mlp_svc_interactions_raw.shape == (SIZE*SIZE, len(original_feature_indices)):
                # ... (vmin/vmax calculation) ...
                vmin_x1_raw, vmax_x1_raw = np.nanpercentile(mlp_svc_interactions_raw[:, 0], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                vmin_x2_raw, vmax_x2_raw = np.nanpercentile(mlp_svc_interactions_raw[:, 1], [PERCENTILE_LOWER, PERCENTILE_UPPER])

                if mlp_svc_interactions_raw[:, 0].shape == true_svc_x1_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, mlp_svc_interactions_raw[:, 0], "SVC_X1_Raw", encoder, "MLP", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                if mlp_svc_interactions_raw[:, 1].shape == true_svc_x2_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, mlp_svc_interactions_raw[:, 1], "SVC_X2_Raw", encoder, "MLP", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                # ... (plotting raw SVC) ...
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
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, mlp_svc_interactions_smooth[:, 0], "SVC_X1_Smooth", encoder, "MLP", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                if mlp_svc_interactions_smooth[:, 1].shape == true_svc_x2_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, mlp_svc_interactions_smooth[:, 1], "SVC_X2_Smooth", encoder, "MLP", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                # ... (plotting smooth SVC) ...
                plot_s([est_intercept_mlp, mlp_svc_interactions_smooth[:, 0], mlp_svc_interactions_smooth[:, 1]],
                       size=SIZE,
                       title=[f"MLP Intercept ({encoder})", f"MLP SVC X1 (Smooth, {encoder})", f"MLP SVC X2 (Smooth, {encoder})"],
                       filename="mlp_svc_smoothed.pdf", experiment_dir=CURRENT_EXPERIMENT_DIR)
        except Exception as e: print(f"Error during MLP GeoShapley explanation or metrics for {encoder}: {type(e).__name__}: {e}")
    else: print(f"Skipping MLP explanation for {encoder} due to training failure.")

    if not xgb_failed:
        print(f"\nExplaining XGBoost model with GeoShapley for {encoder}...")
        try:
            def xgb_predict_func(X_gs_df):
                 if isinstance(X_gs_df, np.ndarray):
                     X_gs_df = pd.DataFrame(X_gs_df, columns=geoshapley_feature_names)
                 X_ml = X_gs_df[ml_feature_names]
                 return xgb_model.predict(X_ml)
            xgb_explainer = GeoShapleyExplainer(xgb_predict_func, background_data_gs.values) # Pass .values
            xgb_rslt = xgb_explainer.explain(explanation_data_gs.values, n_jobs=-1) # Pass .values
            print("XGBoost explanation complete.")
            # ... (summary plot and stats saving) ...
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
                 # ++ Pass coords_for_explained_surfaces ++
                metrics = calculate_spatial_metrics(true_intercept_surface, est_intercept_xgb, "Intercept", encoder, "XGBoost", coords_for_explained_surfaces)
                if metrics: current_model_spatial_metrics.append(metrics)

            xgb_svc_interactions_raw = xgb_rslt.get_svc(col=original_feature_indices, coef_type="raw", include_primary=False)
            if xgb_svc_interactions_raw is not None and xgb_svc_interactions_raw.shape == (SIZE*SIZE, len(original_feature_indices)):
                # ... (vmin/vmax calculation) ...
                vmin_x1_raw_xgb, vmax_x1_raw_xgb = np.nanpercentile(xgb_svc_interactions_raw[:, 0], [PERCENTILE_LOWER, PERCENTILE_UPPER])
                vmin_x2_raw_xgb, vmax_x2_raw_xgb = np.nanpercentile(xgb_svc_interactions_raw[:, 1], [PERCENTILE_LOWER, PERCENTILE_UPPER])

                if xgb_svc_interactions_raw[:, 0].shape == true_svc_x1_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, xgb_svc_interactions_raw[:, 0], "SVC_X1_Raw", encoder, "XGBoost", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                if xgb_svc_interactions_raw[:, 1].shape == true_svc_x2_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, xgb_svc_interactions_raw[:, 1], "SVC_X2_Raw", encoder, "XGBoost", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                # ... (plotting raw SVC) ...
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
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x1_surface, xgb_svc_interactions_smooth[:, 0], "SVC_X1_Smooth", encoder, "XGBoost", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                if xgb_svc_interactions_smooth[:, 1].shape == true_svc_x2_surface.shape:
                     # ++ Pass coords_for_explained_surfaces ++
                    metrics = calculate_spatial_metrics(true_svc_x2_surface, xgb_svc_interactions_smooth[:, 1], "SVC_X2_Smooth", encoder, "XGBoost", coords_for_explained_surfaces)
                    if metrics: current_model_spatial_metrics.append(metrics)
                # ... (plotting smooth SVC) ...
                plot_s([est_intercept_xgb, xgb_svc_interactions_smooth[:, 0], xgb_svc_interactions_smooth[:, 1]],
                       size=SIZE,
                       title=[f"XGB Intercept ({encoder})", f"XGB SVC X1 (Smooth, {encoder})", f"XGB SVC X2 (Smooth, {encoder})"],
                       filename="xgb_svc_smoothed.pdf", experiment_dir=CURRENT_EXPERIMENT_DIR)
        except Exception as e: print(f"Error during XGBoost GeoShapley explanation or metrics for {encoder}: {type(e).__name__}: {e}")
    else: print(f"Skipping XGBoost explanation for {encoder} due to training failure.")

    if current_model_spatial_metrics:
        spatial_metrics_df = pd.DataFrame(current_model_spatial_metrics)
        spatial_metrics_save_path = os.path.join(CURRENT_EXPERIMENT_DIR, "spatial_effect_capture_metrics.csv")
        try:
            spatial_metrics_df.to_csv(spatial_metrics_save_path, index=False)
            print(f"Saved spatial effect capture metrics to {spatial_metrics_save_path}")
            all_spatial_effect_capture_metrics.extend(current_model_spatial_metrics)
        except Exception as e:
            print(f"Error saving spatial effect capture metrics: {e}")
    print(f"===== Finished Encoder: {encoder} =====")

# --- Save Combined Model Performance Metrics ---
print("\nSaving combined model performance metrics across all encoders...")
# ... (rest of the script remains the same)
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
run_mgwr = False # Set to True if you want to run this
if run_mgwr:
    print("\nFitting MGWR model (using original coordinates and X1, X2)...")
    coords_mgwr = original_coords.values
    X_mgwr = X_features.values
    y_mgwr = y.reshape(-1, 1)
    try:
        print("Selecting MGWR bandwidths...")
        # sel = Sel_BW(coords_mgwr, y_mgwr, X_mgwr, multi=True, spherical=False) # Old PySAL
        # bw = sel.search(bw_min=2, multi_bw_min=[2] * X_mgwr.shape[1], verbose=False) # Old PySAL

        # For newer mgwr versions, bandwidth selection might be integrated or different
        # The following is a placeholder if Sel_BW is still used as in your original code
        # from mgwr.sel_bw import Sel_BW # Ensure this is imported if not already
        # sel = Sel_BW(coords_mgwr, y_mgwr, X_mgwr, multi=True)
        # bw = sel.search()
        # print(f"MGWR bandwidths selected: {bw}")
        # mgwr_model = MGWR(coords_mgwr, y_mgwr, X_mgwr, selector=sel, fixed=False, spherical=False)
        # mgwr_results = mgwr_model.fit()


        # --- Simpler MGWR setup if Sel_BW is problematic or for quick test ---
        from mgwr.gwr import MGWR
        from mgwr.sel_bw import Sel_BW

        selector = Sel_BW(coords_mgwr, y_mgwr, X_mgwr, multi=True, spherical=False)
        # You might need to adjust search parameters depending on your data and PySAL version
        # bw = selector.search(multi_bw_min=[2] * (X_mgwr.shape[1] +1), verbose=False) # +1 for intercept
        bw = selector.search(verbose=False)


        print(f"MGWR bandwidths selected: {bw}")
        mgwr_model = MGWR(coords_mgwr, y_mgwr, X_mgwr, selector=selector, fixed=False, spherical=False) # Pass selector object
        mgwr_results = mgwr_model.fit()
        # --- End Simpler MGWR setup ---

        print("MGWR fitting complete.")
        print(mgwr_results.summary())

        if mgwr_results.params.shape[1] == X_mgwr.shape[1] + 1: # Intercept + coeffs
            mgwr_coeffs_to_plot = [mgwr_results.params[:, i] for i in range(mgwr_results.params.shape[1])]
            mgwr_titles = ["MGWR Intercept"] + [f"MGWR Coeff {col}" for col in X_features.columns]
            plot_s(mgwr_coeffs_to_plot, size=SIZE, vmin=1, vmax=5, title=mgwr_titles,
                   filename="mgwr_coefficient_estimates.pdf", experiment_dir=BASE_EXPERIMENT_DIR)

            # Calculate and plot MGWR residuals
            mgwr_residuals = y_mgwr.flatten() - mgwr_results.predy.flatten()
            # Make sure coords_mgwr is (n,2)
            if coords_mgwr.shape[0] == mgwr_residuals.shape[0]:
                try:
                    w_mgwr = weights.Queen.from_array(coords_mgwr)
                    w_mgwr.transform = 'r'
                    moran_mgwr_residuals = Moran(mgwr_residuals, w_mgwr, permutations=99)
                    print(f"\nMGWR Residuals Moran's I: {moran_mgwr_residuals.I:.4f}, P-value: {moran_mgwr_residuals.p_sim:.4f}")
                    # You could save this to a file as well
                    with open(os.path.join(BASE_EXPERIMENT_DIR, "mgwr_moran_i_residuals.txt"), "w") as f:
                        f.write(f"Moran's I: {moran_mgwr_residuals.I}\n")
                        f.write(f"P-value (simulated): {moran_mgwr_residuals.p_sim}\n")
                        f.write(f"Expected I: {moran_mgwr_residuals.EI}\n")
                        f.write(f"Variance (simulated): {moran_mgwr_residuals.VI_sim}\n")
                        f.write(f"Z-value (simulated): {moran_mgwr_residuals.z_sim}\n")

                except Exception as e_moran_mgwr:
                    print(f"Error calculating Moran's I for MGWR residuals: {e_moran_mgwr}")
            else:
                print("Warning: Coordinate and residual count mismatch for MGWR Moran's I calculation.")


        else: print(f"Warning: MGWR params shape {mgwr_results.params.shape} unexpected. Skipping plot.")
        mgwr_summary_path = os.path.join(BASE_EXPERIMENT_DIR, "mgwr_summary.txt")
        with open(mgwr_summary_path, "w") as f: f.write(str(mgwr_results.summary()))
        print(f"Saved MGWR summary: {mgwr_summary_path}")

    except Exception as e: print(f"Error during MGWR analysis: {type(e).__name__}: {e}"); import traceback; traceback.print_exc()
else:
    print("\nSkipping MGWR analysis as run_mgwr is False.")


print("\nScript finished.")