import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import argparse  # New: For command-line arguments

from geoshapley import GeoShapleyExplainer
from help_utils import get_loc_embeddings, plot_s, calculate_spatial_metrics
from sklearn.model_selection import train_test_split
from flaml import AutoML
from mlpSearch import MLP

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Run embedding experiments for a specific encoder."
)
parser.add_argument(
    "--encoder_index", type=int, required=True, help="Index of the encoder type to use."
)
parser.add_argument(
    "--num_repetitions",
    type=int,
    default=1,
    help="Number of times to repeat the experiment for the encoder.",
)
parser.add_argument(
    "--base_experiment_dir",
    type=str,
    default="./results/default",
    help="Base directory for all experiment results.",
)
args = parser.parse_args()

# --- Experiment Setup ---
BASE_EXPERIMENT_DIR = args.base_experiment_dir  # Use from argparse
print(f"Base experiment directory: {BASE_EXPERIMENT_DIR}")
os.makedirs(BASE_EXPERIMENT_DIR, exist_ok=True)

# --- Plotting Configuration ---
SIZE = 25
PERCENTILE_LOWER = 1.0
PERCENTILE_UPPER = 99.0

# --- Spatial Embedding Types ---
# This list should remain complete
all_encoder_types = [
    "Space2Vec-theory",
    "tile_ffn",
    "wrap_ffn",
    "Sphere2Vec-sphereM",
    "Sphere2Vec-sphereM+",
    "rff",
    "Sphere2Vec-sphereC",
    "Sphere2Vec-sphereC+",
    "NeRF",
    "Sphere2Vec-dfs",
    "Space2Vec-grid",
]

# Select the encoder for this specific job run
if args.encoder_index < 0 or args.encoder_index >= len(all_encoder_types):
    print(
        f"FATAL ERROR: Encoder index {args.encoder_index} is out of bounds for {len(all_encoder_types)} encoders."
    )
    exit()
encoder_to_run = all_encoder_types[args.encoder_index]
encoder_types = [encoder_to_run]  # Process only the one for this job array task

print(
    f"Received Encoder Index: {args.encoder_index}, corresponding to Encoder: {encoder_to_run}"
)
print(f"Number of repetitions for this encoder: {args.num_repetitions}")

# --- Data Loading ---
print("Loading data...")
try:
    data_dir = "./data/mgwr_sim.csv"
    mgwr_sim = pd.read_csv(data_dir)
    print(f"Data loaded successfully from {data_dir}.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

original_coords = mgwr_sim[["x_coord", "y_coord"]].copy()
X_features = mgwr_sim[["X1", "X2"]].copy()
y = mgwr_sim.y.values
true_coeffs_df = mgwr_sim[["b0", "b1", "b2"]].copy()

true_intercept_surface = true_coeffs_df["b0"].values.flatten()
true_svc_x1_surface = true_coeffs_df["b1"].values.flatten()
true_svc_x2_surface = true_coeffs_df["b2"].values.flatten()

# Shape check
if not (
    true_intercept_surface.shape == (SIZE * SIZE,)
    and true_svc_x1_surface.shape == (SIZE * SIZE,)
    and true_svc_x2_surface.shape == (SIZE * SIZE,)
):
    print(
        f"FATAL ERROR: True coefficient surfaces do not match expected shape ({SIZE*SIZE},)."
    )
    exit()

# Plot true coefficients (only once, perhaps in a common directory or if encoder_index is 0)
# For simplicity, we'll let each job plot it, but it will overwrite.
# A more robust way would be to check if it exists or do it in a pre-processing step.
plot_s(
    [true_intercept_surface, true_svc_x1_surface, true_svc_x2_surface],
    size=SIZE,
    vmin=1,
    vmax=5,
    title=["True Intercept (b0)", "True Coeff X1 (b1)", "True Coeff X2 (b2)"],
    filename="true_coefficients.pdf",
    experiment_dir=BASE_EXPERIMENT_DIR,  # Save to the main base directory
)


# --- Master Loop for the selected Encoder (but now with repetitions) ---
# Note: The original 'all_model_metrics' and 'all_spatial_effect_capture_metrics'
# were for comparing *across different encoders*.
# Now, we are focusing on repetitions *for a single encoder* per job.
# Global aggregation across encoders will need a separate script after all jobs complete.

coords_for_moran_full = original_coords.values

for (
    encoder
) in encoder_types:  # This loop will run only once for the selected encoder_to_run
    print(f"\n===== Processing Encoder: {encoder} (Repetition Loop Starts) =====")

    # Store metrics and surfaces for each repetition
    repetition_model_metrics_mlp = []
    repetition_model_metrics_xgb = []
    repetition_spatial_metrics_mlp = []
    repetition_spatial_metrics_xgb = []

    # Store estimated surfaces from each repetition to calculate mean/std later
    all_rep_est_intercept_mlp = []
    all_rep_mlp_svc_x1_raw = []
    all_rep_mlp_svc_x2_raw = []
    all_rep_mlp_svc_x1_smooth = []
    all_rep_mlp_svc_x2_smooth = []

    all_rep_est_intercept_xgb = []
    all_rep_xgb_svc_x1_raw = []
    all_rep_xgb_svc_x2_raw = []
    all_rep_xgb_svc_x1_smooth = []
    all_rep_xgb_svc_x2_smooth = []

    for rep_num in range(1, args.num_repetitions + 1):
        print(
            f"\n----- Repetition: {rep_num}/{args.num_repetitions} for Encoder: {encoder} -----"
        )

        # Create a directory for this specific repetition's raw results
        # This is important to avoid overwriting and to keep raw outputs if needed
        REPETITION_DIR = os.path.join(
            BASE_EXPERIMENT_DIR, encoder, f"repetition_{rep_num}"
        )
        os.makedirs(REPETITION_DIR, exist_ok=True)
        print(f"Results for this repetition will be in: {REPETITION_DIR}")

        # Random state for train_test_split should change per repetition for variability
        # Or use a fixed seed for each rep if you want identical splits for debugging,
        # but different seeds across reps for true variability.
        # For statistical variance, different splits are usually desired.
        current_random_state_split = 42 * rep_num  # Simple way to vary seed

        print(f"Generating {encoder} location features...")
        try:
            embeddings = get_loc_embeddings(
                original_coords.values, encoder_type=encoder, device="cpu"
            )  #
            embeddings_np = (
                embeddings.detach().cpu().numpy()
                if isinstance(embeddings, torch.Tensor)
                else np.array(embeddings)
            )
            if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
                print(
                    f"Warning: NaN or Inf values found in embeddings for {encoder} on rep {rep_num}. Skipping this repetition."
                )
                # Add NaN placeholders for this repetition if needed for aggregation later
                continue
            emb_dim = embeddings_np.shape[1]
            emb_cols = [f"{encoder}_emb_{i}" for i in range(emb_dim)]
            X_embeddings = pd.DataFrame(
                embeddings_np, columns=emb_cols, index=original_coords.index
            )
        except Exception as e:
            print(
                f"Error generating embeddings for {encoder} on rep {rep_num}: {e}. Skipping this repetition."
            )
            # Add NaN placeholders
            continue

        X_ml_features = pd.concat([X_features, X_embeddings], axis=1)
        ml_feature_names = X_ml_features.columns.tolist()
        X_for_geoshapley = pd.concat([X_ml_features, original_coords], axis=1)
        geoshapley_feature_names = X_for_geoshapley.columns.tolist()

        print(
            f"\nSplitting data (80/20 split) for repetition {rep_num} with random_state {current_random_state_split}..."
        )
        X_train_gs, X_test_gs, y_train, y_test = train_test_split(
            X_for_geoshapley,
            y,
            test_size=0.20,
            random_state=current_random_state_split,  # Use varying state
        )
        X_train_ml = X_train_gs[ml_feature_names]
        X_test_ml = X_test_gs[ml_feature_names]
        background_data_gs = X_train_gs
        explanation_data_gs = X_for_geoshapley
        coords_for_explained_surfaces = explanation_data_gs[
            ["x_coord", "y_coord"]
        ].values

        current_rep_model_metrics = {}  # For this repetition
        # --- MLP Model ---
        print(f"\nTraining MLP model for {encoder}, repetition {rep_num}...")
        # Use a different random_state for the MLP model in each repetition for model variability
        automl_mlp = AutoML()
        automl_mlp.add_learner(learner_name='mlp', learner_class=MLP)
        mlp_settings = {
            "time_budget": 60,  # in seconds
            "metric": 'r2',
            "task": 'regression',
            "estimator_list": ['mlp'],
            "verbose": 0, # Set to 3 for detailed logs
        }
        mlp_failed = False
        try:
            automl_mlp.fit(X_train=X_train_ml, y_train=y_train, **mlp_settings)
            print(f"  Best MLP learner: {automl_mlp.best_estimator}")
            print(f"  Best MLP config: {automl_mlp.best_config}")
            mlp_model = automl_mlp.model.estimator
            mlp_train_score = mlp_model.score(X_train_ml, y_train)
            mlp_test_score = mlp_model.score(X_test_ml, y_test)
            current_rep_model_metrics["MLP"] = {
                "Train_R2": mlp_train_score,
                "Test_R2": mlp_test_score,
            }
            repetition_model_metrics_mlp.append(
                {
                    "Repetition": rep_num,
                    "Train_R2": mlp_train_score,
                    "Test_R2": mlp_test_score,
                }
            )
        except Exception as e:
            print(f"MLP model training failed for rep {rep_num}: {e}")
            mlp_failed = True
            current_rep_model_metrics["MLP"] = {"Train_R2": np.nan, "Test_R2": np.nan}
            repetition_model_metrics_mlp.append(
                {"Repetition": rep_num, "Train_R2": np.nan, "Test_R2": np.nan}
            )

        # --- XGBoost Model ---
        print(f"\nTraining XGBoost model for {encoder}, repetition {rep_num}...")
        # Use a different random_state for XGBoost in each repetition
        automl_xgb = AutoML()
        xgb_settings = {
            "time_budget": 60,  # in seconds
            "metric": 'r2',
            "task": 'regression',
            "estimator_list": ['xgboost'],
             "verbose": 0,
        }
        xgb_failed = False
        try:
            automl_xgb.fit(X_train=X_train_ml, y_train=y_train, **xgb_settings)
            print(f"  Best XGBoost learner: {automl_xgb.best_estimator}")
            print(f"  Best XGBoost config: {automl_xgb.best_config}")
            xgb_model = automl_xgb.model.estimator
            xgb_train_score = xgb_model.score(X_train_ml, y_train)
            xgb_test_score = xgb_model.score(X_test_ml, y_test)
            current_rep_model_metrics["XGBoost"] = {
                "Train_R2": xgb_train_score,
                "Test_R2": xgb_test_score,
            }
            repetition_model_metrics_xgb.append(
                {
                    "Repetition": rep_num,
                    "Train_R2": xgb_train_score,
                    "Test_R2": xgb_test_score,
                }
            )
        except Exception as e:
            print(f"XGBoost model training failed for rep {rep_num}: {e}")
            xgb_failed = True
            current_rep_model_metrics["XGBoost"] = {
                "Train_R2": np.nan,
                "Test_R2": np.nan,
            }
            repetition_model_metrics_xgb.append(
                {"Repetition": rep_num, "Train_R2": np.nan, "Test_R2": np.nan}
            )

        # Save this repetition's model performance (optional, good for debugging)
        # metrics_df_rep = pd.DataFrame.from_dict(current_rep_model_metrics, orient='index')
        # metrics_df_rep.to_csv(os.path.join(REPETITION_DIR, "model_performance_metrics_rep.csv"))

        original_feature_indices = [
            geoshapley_feature_names.index("X1"),
            geoshapley_feature_names.index("X2"),
        ]

        current_rep_spatial_metrics_mlp = []
        current_rep_spatial_metrics_xgb = []

        # --- MLP GeoShapley Explanation (for this repetition) ---
        if not mlp_failed:
            print(
                f"\nExplaining MLP model with GeoShapley for {encoder}, repetition {rep_num}..."
            )
            try:

                def mlp_predict_func(X_gs_df):
                    if isinstance(X_gs_df, np.ndarray):
                        X_gs_df = pd.DataFrame(
                            X_gs_df, columns=geoshapley_feature_names
                        )
                    X_ml = X_gs_df[ml_feature_names]
                    return mlp_model.predict(X_ml)

                mlp_explainer = GeoShapleyExplainer(
                    mlp_predict_func, background_data_gs.values
                )  #
                mlp_rslt = mlp_explainer.explain(explanation_data_gs, n_jobs=-1)

                # Save summary plot and stats for this repetition (in repetition_X folder)
                try:
                    mlp_rslt.summary_plot(dpi=150)
                    plt.savefig(
                        os.path.join(REPETITION_DIR, "mlp_geoshapley_summary_plot.pdf"),
                        bbox_inches="tight",
                    )
                    plt.close("all")
                    summary_stats_mlp = mlp_rslt.summary_statistics()
                    summary_stats_mlp.to_csv(
                        os.path.join(
                            REPETITION_DIR, "mlp_geoshapley_summary_statistics.csv"
                        )
                    )
                except Exception as e:
                    print(f"Error in MLP summary plot/stats for rep {rep_num}: {e}")
                    plt.close("all")

                est_intercept_mlp = mlp_rslt.base_value + mlp_rslt.geo
                all_rep_est_intercept_mlp.append(
                    est_intercept_mlp
                )  # Store for aggregation
                metrics = calculate_spatial_metrics(
                    true_intercept_surface,
                    est_intercept_mlp,
                    "Intercept",
                    encoder,
                    "MLP",
                    coords_for_explained_surfaces,
                    SIZE,
                )  #
                if metrics:
                    current_rep_spatial_metrics_mlp.append(
                        {**metrics, "Repetition": rep_num}
                    )

                mlp_svc_interactions_raw = mlp_rslt.get_svc(
                    col=original_feature_indices, coef_type="raw", include_primary=False
                )
                if (
                    mlp_svc_interactions_raw is not None
                    and mlp_svc_interactions_raw.shape
                    == (SIZE * SIZE, len(original_feature_indices))
                ):
                    all_rep_mlp_svc_x1_raw.append(mlp_svc_interactions_raw[:, 0])
                    all_rep_mlp_svc_x2_raw.append(mlp_svc_interactions_raw[:, 1])
                    metrics = calculate_spatial_metrics(
                        true_svc_x1_surface,
                        mlp_svc_interactions_raw[:, 0],
                        "SVC_X1_Raw",
                        encoder,
                        "MLP",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_mlp.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    metrics = calculate_spatial_metrics(
                        true_svc_x2_surface,
                        mlp_svc_interactions_raw[:, 1],
                        "SVC_X2_Raw",
                        encoder,
                        "MLP",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_mlp.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    # Plot for this repetition (optional, could save to REPETITION_DIR)
                    # plot_s([est_intercept_mlp, ...], filename="mlp_svc_raw_clipped_rep.pdf", experiment_dir=REPETITION_DIR)

                mlp_svc_interactions_smooth = mlp_rslt.get_svc(
                    col=original_feature_indices, coef_type="gwr", include_primary=False
                )
                if (
                    mlp_svc_interactions_smooth is not None
                    and mlp_svc_interactions_smooth.shape
                    == (SIZE * SIZE, len(original_feature_indices))
                ):
                    all_rep_mlp_svc_x1_smooth.append(mlp_svc_interactions_smooth[:, 0])
                    all_rep_mlp_svc_x2_smooth.append(mlp_svc_interactions_smooth[:, 1])
                    metrics = calculate_spatial_metrics(
                        true_svc_x1_surface,
                        mlp_svc_interactions_smooth[:, 0],
                        "SVC_X1_Smooth",
                        encoder,
                        "MLP",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_mlp.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    metrics = calculate_spatial_metrics(
                        true_svc_x2_surface,
                        mlp_svc_interactions_smooth[:, 1],
                        "SVC_X2_Smooth",
                        encoder,
                        "MLP",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_mlp.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    # Plot for this repetition (optional)
                    # plot_s([est_intercept_mlp, ...], filename="mlp_svc_smoothed_rep.pdf", experiment_dir=REPETITION_DIR)

            except Exception as e:
                print(
                    f"Error during MLP GeoShapley for rep {rep_num}: {type(e).__name__}: {e}"
                )
        repetition_spatial_metrics_mlp.extend(current_rep_spatial_metrics_mlp)

        # --- XGBoost GeoShapley Explanation (for this repetition) ---
        if not xgb_failed:
            print(
                f"\nExplaining XGBoost model with GeoShapley for {encoder}, repetition {rep_num}..."
            )
            try:

                def xgb_predict_func(X_gs_df):
                    if isinstance(X_gs_df, np.ndarray):
                        X_gs_df = pd.DataFrame(
                            X_gs_df, columns=geoshapley_feature_names
                        )
                    X_ml = X_gs_df[ml_feature_names]
                    return xgb_model.predict(X_ml)

                xgb_explainer = GeoShapleyExplainer(
                    xgb_predict_func, background_data_gs.values
                )
                xgb_rslt = xgb_explainer.explain(explanation_data_gs, n_jobs=-1)

                try:
                    xgb_rslt.summary_plot(dpi=150)
                    plt.savefig(
                        os.path.join(REPETITION_DIR, "xgb_geoshapley_summary_plot.pdf"),
                        bbox_inches="tight",
                    )
                    plt.close("all")
                    summary_stats_xgb = xgb_rslt.summary_statistics()
                    summary_stats_xgb.to_csv(
                        os.path.join(
                            REPETITION_DIR, "xgb_geoshapley_summary_statistics.csv"
                        )
                    )
                except Exception as e:
                    print(f"Error in XGBoost summary plot/stats for rep {rep_num}: {e}")
                    plt.close("all")

                est_intercept_xgb = xgb_rslt.base_value + xgb_rslt.geo
                all_rep_est_intercept_xgb.append(est_intercept_xgb)
                metrics = calculate_spatial_metrics(
                    true_intercept_surface,
                    est_intercept_xgb,
                    "Intercept",
                    encoder,
                    "XGBoost",
                    coords_for_explained_surfaces,
                    SIZE,
                )  #
                if metrics:
                    current_rep_spatial_metrics_xgb.append(
                        {**metrics, "Repetition": rep_num}
                    )

                xgb_svc_interactions_raw = xgb_rslt.get_svc(
                    col=original_feature_indices, coef_type="raw", include_primary=False
                )
                if (
                    xgb_svc_interactions_raw is not None
                    and xgb_svc_interactions_raw.shape
                    == (SIZE * SIZE, len(original_feature_indices))
                ):
                    all_rep_xgb_svc_x1_raw.append(xgb_svc_interactions_raw[:, 0])
                    all_rep_xgb_svc_x2_raw.append(xgb_svc_interactions_raw[:, 1])
                    metrics = calculate_spatial_metrics(
                        true_svc_x1_surface,
                        xgb_svc_interactions_raw[:, 0],
                        "SVC_X1_Raw",
                        encoder,
                        "XGBoost",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_xgb.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    metrics = calculate_spatial_metrics(
                        true_svc_x2_surface,
                        xgb_svc_interactions_raw[:, 1],
                        "SVC_X2_Raw",
                        encoder,
                        "XGBoost",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_xgb.append(
                            {**metrics, "Repetition": rep_num}
                        )

                xgb_svc_interactions_smooth = xgb_rslt.get_svc(
                    col=original_feature_indices, coef_type="gwr", include_primary=False
                )
                if (
                    xgb_svc_interactions_smooth is not None
                    and xgb_svc_interactions_smooth.shape
                    == (SIZE * SIZE, len(original_feature_indices))
                ):
                    all_rep_xgb_svc_x1_smooth.append(xgb_svc_interactions_smooth[:, 0])
                    all_rep_xgb_svc_x2_smooth.append(xgb_svc_interactions_smooth[:, 1])
                    metrics = calculate_spatial_metrics(
                        true_svc_x1_surface,
                        xgb_svc_interactions_smooth[:, 0],
                        "SVC_X1_Smooth",
                        encoder,
                        "XGBoost",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_xgb.append(
                            {**metrics, "Repetition": rep_num}
                        )
                    metrics = calculate_spatial_metrics(
                        true_svc_x2_surface,
                        xgb_svc_interactions_smooth[:, 1],
                        "SVC_X2_Smooth",
                        encoder,
                        "XGBoost",
                        coords_for_explained_surfaces,
                        SIZE,
                    )  #
                    if metrics:
                        current_rep_spatial_metrics_xgb.append(
                            {**metrics, "Repetition": rep_num}
                        )
            except Exception as e:
                print(
                    f"Error during XGBoost GeoShapley for rep {rep_num}: {type(e).__name__}: {e}"
                )
        repetition_spatial_metrics_xgb.extend(current_rep_spatial_metrics_xgb)

        print(
            f"----- Finished Repetition: {rep_num}/{args.num_repetitions} for Encoder: {encoder} -----"
        )

    # --- Aggregation and Saving for the current ENCODER across all its repetitions ---
    ENCODER_RESULTS_DIR = os.path.join(
        BASE_EXPERIMENT_DIR, encoder
    )  # Main folder for this encoder's aggregated results
    os.makedirs(ENCODER_RESULTS_DIR, exist_ok=True)

    # Aggregate Model Performance Metrics
    if repetition_model_metrics_mlp:
        mlp_metrics_df = pd.DataFrame(repetition_model_metrics_mlp)
        mlp_metrics_summary = mlp_metrics_df.drop(columns=["Repetition"]).agg(
            ["mean", "std", "count"]
        )
        mlp_metrics_df.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "mlp_model_performance_all_reps.csv"),
            index=False,
        )
        mlp_metrics_summary.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "mlp_model_performance_summary.csv")
        )
        print(f"\nMLP Model Performance Summary for {encoder}:\n{mlp_metrics_summary}")

    if repetition_model_metrics_xgb:
        xgb_metrics_df = pd.DataFrame(repetition_model_metrics_xgb)
        xgb_metrics_summary = xgb_metrics_df.drop(columns=["Repetition"]).agg(
            ["mean", "std", "count"]
        )
        xgb_metrics_df.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "xgb_model_performance_all_reps.csv"),
            index=False,
        )
        xgb_metrics_summary.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "xgb_model_performance_summary.csv")
        )
        print(
            f"\nXGBoost Model Performance Summary for {encoder}:\n{xgb_metrics_summary}"
        )

    # Aggregate Spatial Effect Capture Metrics
    if repetition_spatial_metrics_mlp:
        mlp_spatial_df = pd.DataFrame(repetition_spatial_metrics_mlp)
        # Group by spatial_effect and then aggregate
        mlp_spatial_summary = (
            mlp_spatial_df.drop(columns=["Repetition", "encoder", "model"])
            .groupby("spatial_effect")
            .agg(["mean", "std", "count"])
        )
        mlp_spatial_df.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "mlp_spatial_metrics_all_reps.csv"),
            index=False,
        )
        mlp_spatial_summary.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "mlp_spatial_metrics_summary.csv")
        )
        print(f"\nMLP Spatial Metrics Summary for {encoder}:\n{mlp_spatial_summary}")

    if repetition_spatial_metrics_xgb:
        xgb_spatial_df = pd.DataFrame(repetition_spatial_metrics_xgb)
        xgb_spatial_summary = (
            xgb_spatial_df.drop(columns=["Repetition", "encoder", "model"])
            .groupby("spatial_effect")
            .agg(["mean", "std", "count"])
        )
        xgb_spatial_df.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "xgb_spatial_metrics_all_reps.csv"),
            index=False,
        )
        xgb_spatial_summary.to_csv(
            os.path.join(ENCODER_RESULTS_DIR, "xgb_spatial_metrics_summary.csv")
        )
        print(
            f"\nXGBoost Spatial Metrics Summary for {encoder}:\n{xgb_spatial_summary}"
        )

    # --- Calculate Mean and Std Dev of Estimated Surfaces & Plot ---
    # Define a helper for robust mean/std calculation and plotting
    def aggregate_and_plot_surfaces(
        surface_list, true_surface, base_title, filename_prefix, model_name
    ):
        if not surface_list:
            print(f"No surfaces collected for {model_name} {base_title} to aggregate.")
            return

        # Ensure all surfaces are numpy arrays and handle potential NaNs from failed reps
        processed_surfaces = []
        for s in surface_list:
            if isinstance(s, np.ndarray):
                processed_surfaces.append(s)
            # else: print(f"Warning: Invalid surface type in list for {base_title}")

        if not processed_surfaces:
            print(f"No valid numpy array surfaces for {model_name} {base_title}.")
            return

        try:
            surfaces_stack = np.array(
                processed_surfaces
            )  # Stack along a new axis (repetitions)
            mean_surface = np.nanmean(surfaces_stack, axis=0)
            std_surface = np.nanstd(surfaces_stack, axis=0)

            # Save mean and std surfaces as .npy files for later use if needed
            np.save(
                os.path.join(
                    ENCODER_RESULTS_DIR, f"{filename_prefix}_mean_surface.npy"
                ),
                mean_surface,
            )
            np.save(
                os.path.join(ENCODER_RESULTS_DIR, f"{filename_prefix}_std_surface.npy"),
                std_surface,
            )

            # Plotting
            # You might need a new plotting function in help_utils.py
            # that can take mean and std_dev surfaces, or plot them side-by-side.
            # For now, let's plot them separately.
            plot_s(
                [mean_surface, std_surface, true_surface],  #
                size=SIZE,
                title=[
                    f"Mean {base_title} ({model_name}, {encoder})",
                    f"Std Dev {base_title} ({model_name}, {encoder})",
                    f"True {base_title}",
                ],
                filename=f"{filename_prefix}_mean_std_true.pdf",
                experiment_dir=ENCODER_RESULTS_DIR,
            )

        except Exception as e:
            print(
                f"Error aggregating/plotting surfaces for {model_name} {base_title}: {e}"
            )
            import traceback

            traceback.print_exc()

    # MLP Surfaces
    aggregate_and_plot_surfaces(
        all_rep_est_intercept_mlp,
        true_intercept_surface,
        "Intercept",
        f"mlp_{encoder}_intercept",
        "MLP",
    )
    aggregate_and_plot_surfaces(
        all_rep_mlp_svc_x1_raw,
        true_svc_x1_surface,
        "SVC X1 (Raw)",
        f"mlp_{encoder}_svc_x1_raw",
        "MLP",
    )
    aggregate_and_plot_surfaces(
        all_rep_mlp_svc_x2_raw,
        true_svc_x2_surface,
        "SVC X2 (Raw)",
        f"mlp_{encoder}_svc_x2_raw",
        "MLP",
    )
    aggregate_and_plot_surfaces(
        all_rep_mlp_svc_x1_smooth,
        true_svc_x1_surface,
        "SVC X1 (Smooth)",
        f"mlp_{encoder}_svc_x1_smooth",
        "MLP",
    )
    aggregate_and_plot_surfaces(
        all_rep_mlp_svc_x2_smooth,
        true_svc_x2_surface,
        "SVC X2 (Smooth)",
        f"mlp_{encoder}_svc_x2_smooth",
        "MLP",
    )

    # XGBoost Surfaces
    aggregate_and_plot_surfaces(
        all_rep_est_intercept_xgb,
        true_intercept_surface,
        "Intercept",
        f"xgb_{encoder}_intercept",
        "XGBoost",
    )
    aggregate_and_plot_surfaces(
        all_rep_xgb_svc_x1_raw,
        true_svc_x1_surface,
        "SVC X1 (Raw)",
        f"xgb_{encoder}_svc_x1_raw",
        "XGBoost",
    )
    aggregate_and_plot_surfaces(
        all_rep_xgb_svc_x2_raw,
        true_svc_x2_surface,
        "SVC X2 (Raw)",
        f"xgb_{encoder}_svc_x2_raw",
        "XGBoost",
    )
    aggregate_and_plot_surfaces(
        all_rep_xgb_svc_x1_smooth,
        true_svc_x1_surface,
        "SVC X1 (Smooth)",
        f"xgb_{encoder}_svc_x1_smooth",
        "XGBoost",
    )
    aggregate_and_plot_surfaces(
        all_rep_xgb_svc_x2_smooth,
        true_svc_x2_surface,
        "SVC X2 (Smooth)",
        f"xgb_{encoder}_svc_x2_smooth",
        "XGBoost",
    )

    print(f"===== Finished All Repetitions for Encoder: {encoder} =====")

# Note: MGWR analysis and combined metrics across ALL encoders are removed from this script.
# The MGWR part (if needed) should be run separately, as it's not encoder-specific in the same way.
# Combined metrics across all encoders would require a new post-processing script
# that reads the summary CSVs from each encoder's subfolder in BASE_EXPERIMENT_DIR.

print(f"\nScript finished for Encoder: {encoder_to_run} (Index: {args.encoder_index}).")
