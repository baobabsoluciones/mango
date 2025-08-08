#!/usr/bin/env python
"""
Train an AutoEncoder model on specific wind farm data.

This script loads data from Aberdeen and Loxton processed datasets
and trains an autoencoder model for time series imputation.
"""

import logging
from pathlib import Path

import pandas as pd
from mango_autoencoder import AutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_data_files(file_paths):
    """
    Load and combine data from multiple CSV files.

    :param file_paths: List of paths to CSV files
    :type file_paths: list
    :return: Combined DataFrame with source file information
    :rtype: pd.DataFrame
    """
    dfs = []

    for idx, fp in enumerate(file_paths):
        logger.info(f"Loading file {idx+1}/{len(file_paths)}: {fp}")
        temp_df = pd.read_csv(fp)

        file_name = Path(fp).stem
        temp_df["source_file"] = file_name

        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data shape: {df.shape}")

    return df


def train_autoencoder(data):
    """
    Train an AutoEncoder model on the provided data.

    :param data: DataFrame containing the data
    :type data: pd.DataFrame
    :return: Trained AutoEncoder model
    :rtype: AutoEncoder
    """
    output_dir = Path("autoencoder_output/wind_farms")
    output_dir.mkdir(parents=True, exist_ok=True)

    # if nunique in source_file is 1
    if data["source_file"].nunique() == 1:
        data.drop(columns=["source_file"], inplace=True)

    if "source_file" in data.columns:
        id_columns = "source_file" if data["source_file"].nunique() > 1 else None
    else:
        id_columns = None

    model = AutoEncoder()
    model.build_model(
        form="lstm",
        data=data,
        context_window=10,
        id_columns=id_columns,
        time_step_to_check=[9],
        hidden_dim=[8, 4],
        feature_to_check=list(range(min(8, len(data.columns)))),
        bidirectional_encoder=True,
        bidirectional_decoder=False,
        normalize=True,
        normalization_method="minmax",
        batch_size=32,
        save_path=str(output_dir),
        verbose=True,
        use_mask=True,
        shuffle=True,
    )

    model.train(epochs=1, use_early_stopping=True, checkpoint=2)
    model.reconstruct()

    return model


def main():
    """
    Main execution function to load data and train the model.
    """
    file_paths = [
        r"G:\Unidades compartidas\mango\desarrollo\datos\autoencoders\imputation\acciona_velocidad\processed_aberdeen_1.csv",
        r"G:\Unidades compartidas\mango\desarrollo\datos\autoencoders\imputation\acciona_velocidad\processed_loxton_1.csv",
    ]

    logger.info("Loading data from specified files")
    data = load_data_files(file_paths)

    logger.info("Training autoencoder model")
    model = train_autoencoder(data)

    logger.info(
        f"Model training completed. Results saved in 'autoencoder_output/wind_farms'"
    )

    # Extract and reconstruct the Loxton dataset specifically
    logger.info("Reconstructing Loxton dataset")
    # loxton_data = data[data["source_file"] == "processed_loxton_1"].copy()
    # loxton_data.drop(columns=["source_file"], inplace=True)

    # Create directory for reconstruction results
    reconstruct_output_dir = Path("autoencoder_output/wind_farms/loxton_reconstruction")
    reconstruct_output_dir.mkdir(parents=True, exist_ok=True)

    # Perform reconstruction on the Loxton dataset
    reconstructed_results = model.reconstruct_new_data(id_columns="source_file",
        data=data, iterations=3, save_path=str(reconstruct_output_dir)
    )

    # Save reconstructed data to CSV
    for id_key, df in reconstructed_results.items():
        output_file = reconstruct_output_dir / f"reconstructed_{id_key}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Reconstructed data saved to {output_file}")


if __name__ == "__main__":
    main()
