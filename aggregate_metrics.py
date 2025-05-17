import pandas as pd
import os
import glob

# --- Configuration ---
BASE_RESULTS_DIR = (
    "./results/dumb_multi_embedding_fixed2_repeated"  # Must match embeddingsRun.py
)
OUTPUT_DIR = BASE_RESULTS_DIR  # Save combined files in the base results directory
ENCODER_DIRS = [
    d for d in glob.glob(os.path.join(BASE_RESULTS_DIR, "*")) if os.path.isdir(d)
]

# --- File Names (must match what embeddingsRun.py saves) ---
MLP_MODEL_SUMMARY_FILENAME = "mlp_model_performance_summary.csv"
XGB_MODEL_SUMMARY_FILENAME = "xgb_model_performance_summary.csv"
MLP_SPATIAL_SUMMARY_FILENAME = "mlp_spatial_metrics_summary.csv"
XGB_SPATIAL_SUMMARY_FILENAME = "xgb_spatial_metrics_summary.csv"

all_mlp_model_summaries = []
all_xgb_model_summaries = []
all_mlp_spatial_summaries = []
all_xgb_spatial_summaries = []

print(f"Aggregating results from: {BASE_RESULTS_DIR}")
print(f"Found encoder directories: {ENCODER_DIRS}")

for encoder_dir in ENCODER_DIRS:
    encoder_name = os.path.basename(encoder_dir)
    print(f"... Processing {encoder_name}")

    # MLP Model Performance
    mlp_model_file = os.path.join(encoder_dir, MLP_MODEL_SUMMARY_FILENAME)
    if os.path.exists(mlp_model_file):
        try:
            df = pd.read_csv(mlp_model_file)
            df["encoder"] = encoder_name
            # The summary file has 'mean', 'std', 'count' as index.
            # We might want to flatten this or select specific rows (e.g., mean)
            # For simplicity, let's take the mean row if available, or adjust as needed.
            # Assuming the first column is the statistic type (mean, std)
            df = df.rename(columns={"Unnamed: 0": "statistic_type"})
            df_mean = df[df["statistic_type"] == "mean"].copy()
            df_mean["model"] = "MLP"
            all_mlp_model_summaries.append(df_mean)
        except Exception as e:
            print(f"Error reading {mlp_model_file}: {e}")

    # XGBoost Model Performance
    xgb_model_file = os.path.join(encoder_dir, XGB_MODEL_SUMMARY_FILENAME)
    if os.path.exists(xgb_model_file):
        try:
            df = pd.read_csv(xgb_model_file)
            df["encoder"] = encoder_name
            df = df.rename(columns={"Unnamed: 0": "statistic_type"})
            df_mean = df[df["statistic_type"] == "mean"].copy()
            df_mean["model"] = "XGBoost"
            all_xgb_model_summaries.append(df_mean)
        except Exception as e:
            print(f"Error reading {xgb_model_file}: {e}")

    # MLP Spatial Metrics
    mlp_spatial_file = os.path.join(encoder_dir, MLP_SPATIAL_SUMMARY_FILENAME)
    if os.path.exists(mlp_spatial_file):
        try:
            df = pd.read_csv(
                mlp_spatial_file
            )  # This file has 'spatial_effect' as a column
            df["encoder"] = encoder_name
            # The summary has multi-index (spatial_effect, statistic_type).
            # We can melt or pivot this for a flatter structure.
            # For now, let's load and select 'mean' rows.
            # This might need more sophisticated reshaping depending on desired output.
            # Assuming first column is spatial_effect, second is statistic_type
            # This depends on how it's saved. If it's already grouped, pandas reads it with multi-index.
            # Let's assume it was saved with reset_index() before saving or read carefully.
            # Simpler: read and filter by the level of multi-index if needed, or reconstruct.
            # For now, just append and you can reshape later.
            df_mean_spatial = df[
                df.iloc[:, 1] == "mean"
            ].copy()  # Assuming second col is stat_type after groupby
            df_mean_spatial["model"] = "MLP"
            all_mlp_spatial_summaries.append(df_mean_spatial)
        except Exception as e:
            print(f"Error reading {mlp_spatial_file}: {e}")

    # XGBoost Spatial Metrics
    xgb_spatial_file = os.path.join(encoder_dir, XGB_SPATIAL_SUMMARY_FILENAME)
    if os.path.exists(xgb_spatial_file):
        try:
            df = pd.read_csv(xgb_spatial_file)
            df["encoder"] = encoder_name
            df_mean_spatial = df[df.iloc[:, 1] == "mean"].copy()
            df_mean_spatial["model"] = "XGBoost"
            all_xgb_spatial_summaries.append(df_mean_spatial)
        except Exception as e:
            print(f"Error reading {xgb_spatial_file}: {e}")


# Combine and Save
if all_mlp_model_summaries or all_xgb_model_summaries:
    all_model_perf_df = pd.concat(
        all_mlp_model_summaries + all_xgb_model_summaries, ignore_index=True
    )
    if not all_model_perf_df.empty:
        # Reorder columns for clarity
        cols_order = ["encoder", "model", "statistic_type"] + [
            c
            for c in all_model_perf_df.columns
            if c not in ["encoder", "model", "statistic_type"]
        ]
        all_model_perf_df = all_model_perf_df[cols_order]
        all_model_perf_df.to_csv(
            os.path.join(OUTPUT_DIR, "all_encoders_model_performance_mean_summary.csv"),
            index=False,
        )
        print(
            f"\nSaved combined model performance (mean) summary to {os.path.join(OUTPUT_DIR, 'all_encoders_model_performance_mean_summary.csv')}"
        )
        print(all_model_perf_df.head())

if all_mlp_spatial_summaries or all_xgb_spatial_summaries:
    all_spatial_perf_df = pd.concat(
        all_mlp_spatial_summaries + all_xgb_spatial_summaries, ignore_index=True
    )
    if not all_spatial_perf_df.empty:
        # This needs careful column ordering based on how the spatial summary was saved and read
        # Example: renaming the first column to 'spatial_effect' and second to 'statistic_type'
        # if they were part of a multi-index before saving.
        # This part may need adjustment based on the exact CSV structure.
        # For now, assume 'spatial_effect' is the first column from the read CSV after filtering by 'mean'.
        try:
            all_spatial_perf_df = all_spatial_perf_df.rename(
                columns={
                    all_spatial_perf_df.columns[0]: "spatial_effect",
                    all_spatial_perf_df.columns[1]: "statistic_type",
                }
            )
            cols_order = ["encoder", "model", "spatial_effect", "statistic_type"] + [
                c
                for c in all_spatial_perf_df.columns
                if c not in ["encoder", "model", "spatial_effect", "statistic_type"]
            ]
            all_spatial_perf_df = all_spatial_perf_df[cols_order]
        except Exception as e:
            print(f"Warning: Could not reorder columns for spatial summary: {e}")

        all_spatial_perf_df.to_csv(
            os.path.join(OUTPUT_DIR, "all_encoders_spatial_metrics_mean_summary.csv"),
            index=False,
        )
        print(
            f"\nSaved combined spatial metrics (mean) summary to {os.path.join(OUTPUT_DIR, 'all_encoders_spatial_metrics_mean_summary.csv')}"
        )
        print(all_spatial_perf_df.head())

print("\nAggregation script finished.")
