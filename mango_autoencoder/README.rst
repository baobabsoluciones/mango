Mango Autoencoder
<<<<<<< HEAD
================
=======
=================
>>>>>>> master

A Python library for anomaly detection in time series using neural autoencoders.

Description
-----------

Mango Autoencoder is a specialized tool for time series analysis that uses neural autoencoder networks to detect anomalies and reconstruct data. It is designed to be highly configurable and easy to use, with advanced data processing capabilities.

Key Features
------------

**Flexible Neural Architectures**
- Supports LSTM, GRU, and RNN

**Anomaly Detection**
- Automatic identification of anomalous patterns in time series

**Data Reconstruction**
- Ability to reconstruct missing or corrupted data

**Advanced Processing**
- Normalization, imputation, and handling of missing values

**Integrated Visualization**
- Plotting tools for result analysis

**Bidirectional Configuration**
- Support for bidirectional encoders and decoders

**Mask Handling**
- Intelligent data processing with custom masks

**New Data Reconstruction**
- Reconstruct unknown data with iterative improvement

Installation
------------

**Using uv:**

.. code-block:: bash

   # Create virtual environment with Python 3.11
   uv venv --python 3.11
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   uv add mango-autoencoder

**Using pip:**

.. code-block:: bash

   pip install mango-autoencoder

Dependencies
------------

- Python >= 3.10
- TensorFlow >= 2.18.0
- Pandas >= 2.0.3
- Polars >= 1.31.0
- Scikit-learn >= 1.6.1
- Plotly >= 6.2.0

Basic Usage
-----------

.. code-block:: python

   from mango_autoencoder import AutoEncoder
   import numpy as np

   # Create autoencoder instance
   autoencoder = AutoEncoder()

   # Configure and train the model
   autoencoder.build_and_train(
       context_window=10,
       data=time_series_data,
       time_step_to_check=[0, 1, 2],
       feature_to_check=[0, 1],
       hidden_dim=64,
       form="lstm",
       epochs=100
   )

   # Reconstruct data
   reconstruction = autoencoder.reconstruct()

Advanced Usage: Reconstructing New Data
---------------------------------------

The ``reconstruct_new_data`` method allows you to reconstruct unknown data using a trained model. This is particularly useful for:

- **Missing Data Imputation**: Fill in missing values in time series
- **Data Quality Improvement**: Correct corrupted or noisy data
- **Iterative Refinement**: Improve reconstruction quality through multiple iterations

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from mango_autoencoder import AutoEncoder

   # Load a trained model
   model = AutoEncoder.load_from_pickle("path/to/model.pkl")

   # Set up output directory
   reconstruct_output_dir = Path("autoencoder_output/reconstruction")
   reconstruct_output_dir.mkdir(parents=True, exist_ok=True)

   # Perform reconstruction on new data
   reconstructed_results = model.reconstruct_new_data(
       id_columns="source_file",
       data=data,
       iterations=3,
       save_path=str(reconstruct_output_dir),
       reconstruction_diagnostic=True
   )

Parameters
~~~~~~~~~~

- **``data``**: Input data (numpy array, pandas DataFrame, or polars DataFrame)
- **``iterations``**: Number of reconstruction iterations (default: 1)
  - Higher iterations can improve reconstruction quality for data with many missing values
  - Each iteration uses the previous reconstruction to improve the next one
- **``id_columns``**: Column(s) that define IDs to process reconstruction separately
  - Useful when data contains multiple time series (e.g., different sensors, locations)
  - Can be a string, integer, or list of strings/integers
- **``save_path``**: Path to save reconstruction results and diagnostics
- **``reconstruction_diagnostic``**: If True, generates error analysis and visualization files

How It Works
~~~~~~~~~~~~

1. **Data Validation**: Checks that the new data has the same features as the training data
2. **ID Processing**: Separates data by ID columns if specified
3. **Iterative Reconstruction**:
   - For each iteration, the model reconstructs the data
   - Missing values (NaN) are filled with reconstructed values
   - The process repeats to improve reconstruction quality
4. **Result Generation**: Returns reconstructed data and optionally saves diagnostic files

Output Files
------------

Training Phase
~~~~~~~~~~~~~~

When you train a model with ``build_and_train()``, the following files are created in the specified ``save_path``:

Model Files
~~~~~~~~~~~

- **``models/model.pkl``**: Main model file containing the trained Keras model and training parameters
- **``models/{epoch}.pkl``**: Checkpoint files saved every ``checkpoint`` epochs (e.g., ``10.pkl``, ``20.pkl``)

Visualization Files
~~~~~~~~~~~~~~~~~~~

- **``loss_history.html``**: Interactive plot showing training and validation loss over epochs

Reconstruction Files (if ``reconstruction_diagnostic=True``)
<<<<<<< HEAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=======
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>>>>>> master

- **``actual_vs_reconstructed.html``**: Interactive plot comparing original vs reconstructed data
- **``reconstruction_error.csv``**: Detailed reconstruction error data
- **``reconstruction_error_summary.csv``**: Summary statistics of reconstruction errors
- **``reconstruction_error_boxplot.html``**: Box plot visualization of reconstruction errors by feature and data split

Reconstruction Phase (``reconstruct_new_data``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``reconstruct_new_data()``, the following files are created in the specified ``save_path``:

Reconstruction Results
~~~~~~~~~~~~~~~~~~~~~~

- **``reconstruct_new_data/{id}_reconstruction_results.csv``**: Reconstructed data for each ID (or "global" if no IDs)

Diagnostic Files (if ``reconstruction_diagnostic=True``)
<<<<<<< HEAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=======
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>>>>>> master

- **``reconstruct_new_data/{id}_reconstruction_error.csv``**: Reconstruction error data for each ID
- **``reconstruct_new_data/{id}_reconstruction_error_summary.csv``**: Summary statistics for each ID
- **``reconstruct_new_data/{id}_reconstruction_error_boxplot.html``**: Box plot of reconstruction errors for each ID

File Structure Example
~~~~~~~~~~~~~~~~~~~~~~

::

   autoencoder_output/
   ├── models/
   │   ├── model.pkl
   │   ├── 10.pkl
   │   ├── 20.pkl
   │   └── ...
   ├── loss_history.html
   ├── actual_vs_reconstructed.html
   ├── reconstruction_error.csv
   ├── reconstruction_error_summary.csv
   ├── reconstruction_error_boxplot.html
   └── reconstruct_new_data/
       ├── global_reconstruction_results.csv
       ├── global_reconstruction_error.csv
       ├── global_reconstruction_error_summary.csv
       └── global_reconstruction_error_boxplot.html

Project Structure
-----------------

::

   mango_autoencoder/
   ├── mango_autoencoder/
   │   ├── autoencoder.py          # Main autoencoder class
   │   ├── modules/
   │   │   ├── encoder.py          # Encoding module
   │   │   ├── decoder.py          # Decoding module
   │   │   └── anomaly_detector.py # Anomaly detector
   │   ├── utils/
   │   │   ├── processing.py       # Processing utilities
   │   │   ├── plots.py           # Visualization tools
   │   │   └── sequences.py       # Sequence processing
   │   ├── tests/                  # Unit tests
   │   │   └── test_autoencoder.py # Autoencoder tests
   │   └── logging/                # Logging utilities
   ├── pyproject.toml             # Project configuration
   └── uv.lock                    # Dependency lock file

Documentation
-------------

For detailed documentation, visit the `Mango Documentation <https://baobabsoluciones.github.io/mango/>`_.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <https://baobabsoluciones.es/>`_