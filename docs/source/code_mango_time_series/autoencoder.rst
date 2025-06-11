Autoencoder
===========

.. note::
    This documentation is still under development. If you find any bug or have any suggestion in the decodoc module, please, open an issue in the `GitHub repository <https://github.com/baobabsoluciones/mango>`_.

The autoencoder module provides a powerful implementation of autoencoder neural networks specifically designed for time series data. This implementation supports various architectures and configurations for encoding and decoding time series data, making it suitable for tasks like anomaly detection, data reconstruction, and feature learning.

Overview
~~~~~~~~

The autoencoder consists of three main components:

- **Encoder**: Compresses the input time series data into a lower-dimensional representation.
- **Decoder**: Reconstructs the original data from the compressed representation.
- **Training pipeline**: Handles data preprocessing, model training, and evaluation.

The implementation supports multiple neural network architectures:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- RNN (Simple Recurrent Neural Network)
- Dense (Fully Connected Neural Network) - Note: Not implemented for time series autoencoders

Theoretical background
~~~~~~~~~~~~~~~~~~~~

Autoencoders are a class of unsupervised neural networks that learn to compress and reconstruct input data. Their core objective is to encode the input into a latent representation that captures the most relevant information, and then decode it to produce a reconstruction as close as possible to the original input. The loss function—typically Mean Squared Error (MSE)—guides the learning process by penalizing discrepancies between the input and its reconstruction.

In the context of time series, choosing the appropriate network architecture is critical due to the sequential and often variable-length nature of the data. This implementation supports three recurrent neural network (RNN) architectures, each suited to different characteristics of time series dynamics.

**Long Short-Term Memory (LSTM)** networks are designed to capture long-term dependencies by maintaining a cell state that is modulated through input, forget, and output gates. This makes them well suited for datasets where patterns or influences persist over many time steps.

**Gated Recurrent Units (GRU)** offer a simplified version of LSTM with fewer gates and parameters, often resulting in faster training and comparable performance. GRUs are effective when training time or computational resources are limited, or when the dataset does not require the full complexity of an LSTM.

**Simple RNNs** are the most basic recurrent architecture. While easier to implement and interpret, they often suffer from vanishing or exploding gradients when processing long sequences, making them more suitable for shorter-term dependencies.

Although **dense (fully connected) layers** are also available in this module, they are not designed for time series autoencoders. Dense layers operate on fixed-size input vectors and lack temporal memory, meaning they cannot effectively model sequential dependencies unless paired with specific preprocessing or architectural adaptations. In this implementation, dense layers are supported only as post-processing layers or for non-sequential encoders and decoders.

Currently, this implementation specifically supports several key applications while providing a foundation for others:

- **Missing data reconstruction**: This is one of the primary implemented features of this autoencoder. The model can intelligently reconstruct missing values in time series data by learning the underlying patterns from the available data. This capability is particularly valuable for sensor data with intermittent failures, financial time series with missing trading days, or environmental data with measurement gaps. The `use_mask` parameter allows you to specify which values are missing, and the model will focus on reconstructing those specific points. This approach is more sophisticated than simple interpolation methods, as it considers the entire context of the time series rather than just neighboring values.

- **Data compression and feature extraction**: The encoder component of this autoencoder compresses time series data into a lower-dimensional representation while preserving the most important patterns. This compressed representation can be used as features for other machine learning tasks, reducing dimensionality while maintaining essential information. This is particularly useful for high-dimensional time series data where dimensionality reduction is critical.

- **Noise reduction and smoothing**: This implementation can effectively filter out noise from time series data. By forcing the network to reconstruct the original data from a compressed representation, it learns to ignore random fluctuations and focus on the underlying patterns. This results in smoother, more interpretable time series with reduced impact of measurement errors. The `normalize` parameter can help standardize the data before processing, further enhancing the noise reduction capabilities.

- **Anomaly detection** (in development): This feature is currently in development and not yet fully implemented. The autoencoder architecture has the potential to detect anomalies in time series data by identifying patterns that deviate from the learned normal behavior. When this feature is completed, it will allow users to identify unusual patterns or outliers in time series data, which is particularly useful for identifying system failures, unusual trading patterns, or process deviations.

Other potential applications that are not currently implemented include:

- **Time series forecasting**: While this autoencoder can learn patterns in time series data, it does not directly implement forecasting functionality. The context window parameter is used for sequence processing during training and reconstruction, not for making future predictions. For forecasting tasks, consider using specialized forecasting models or using the autoencoder's compressed representation as input features for a forecasting model.

- **Change point detection**: Autoencoders can be used to detect significant changes in time series patterns, such as regime shifts or structural breaks. While not directly implemented in this module, the reconstruction error patterns could be analyzed to identify potential change points.

- **Dimensionality reduction for visualization**: The compressed representation from the encoder can be used for visualizing high-dimensional time series data in lower dimensions (e.g., 2D or 3D) for exploratory analysis. This application is supported indirectly through the feature extraction capabilities.

- **Transfer learning**: The learned representations from this autoencoder could be transferred to other related time series tasks, though this would require additional implementation beyond the current module.

- **Multi-variate time series analysis**: While this implementation supports multi-variate time series, specialized applications like cross-series dependency analysis would require additional implementation.

Architecture
~~~~~~~~~~~

The autoencoder architecture is highly configurable through the following components:

**Encoder**

The encoder compresses the input time series data into a lower-dimensional representation. Available architectures:

- **LSTM Encoder**: Uses Long Short-Term Memory layers for capturing long-term dependencies
- **GRU Encoder**: Uses Gated Recurrent Unit layers for efficient sequence processing
- **RNN Encoder**: Uses Simple RNN layers for basic sequence processing
- **Dense Encoder**: Uses fully connected layers for non-sequential data

.. note::
    **Dense architecture in autoencoders**:
    
    The Dense architecture is available for individual encoders and decoders, but **not implemented for time series autoencoders** because:
    
    1. Time series data is inherently sequential and variable-length
    2. Dense layers require fixed-size input tensors
    3. Dense layers cannot capture temporal dependencies between time steps
    
    For time series autoencoders, use LSTM, GRU, or RNN architectures instead.

**Decoder**

The decoder reconstructs the original data from the compressed representation. Available architectures:

- **LSTM Decoder**: Reconstructs sequences using LSTM layers
- **GRU Decoder**: Reconstructs sequences using GRU layers
- **RNN Decoder**: Reconstructs sequences using Simple RNN layers
- **Dense Decoder**: Reconstructs data using fully connected layers

**Utils module**

The utils module provides a collection of utility functions and tools for preprocessing, normalizing, and visualizing time series data in the context of autoencoder models.

The utils module is organized into several submodules:

- **Processing**: Data preprocessing and transformation functions
- **Plots**: Visualization tools for model evaluation and analysis
- **Sequences**: Time series sequence handling utilities

Configuration and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The AutoEncoder class provides extensive configuration options through its parameters. Here's a detailed explanation of each parameter and its functionality:

**Required parameters**

The following parameters are mandatory when calling `build_model` or `build_and_train`:

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **context_window**
     - Size of the context window for sequence transformation.
       
       This is a crucial parameter that determines how the time series data is processed. It defines the number of consecutive time steps that will be grouped together to form a sequence. For example, if context_window=10, each input sequence will contain 10 consecutive time steps.
       
       The context window transforms your 2D data (samples × features) into 3D data (samples × context_window × features). This transformation is essential for recurrent neural networks (LSTM, GRU, RNN) to process sequential patterns.
       
       A larger context window allows the model to capture longer-term dependencies but requires more memory and computation. A smaller context window is more efficient but may miss long-term patterns. The optimal context window depends on your specific time series characteristics and the temporal patterns you want to capture.
   * - **data**
     - Input data for training. Can be provided in two formats:
       
       **Single dataset format**: A single DataFrame/array containing all your time series data. In this case, the autoencoder will automatically split the data into train, validation, and test sets. The split proportions are controlled by the train_size, val_size, and test_size parameters. This is the simplest approach when you have a single dataset and want automatic splitting.
       
       **Pre-split format**: A tuple of three arrays (train_data, val_data, test_data). In this case, you provide the data already split into training, validation, and test sets. The autoencoder will use these pre-split datasets without performing any additional splitting. This gives you full control over how the data is divided and is useful when you have specific splitting requirements. The train_size, val_size, and test_size parameters are ignored when using this format.
   * - **time_step_to_check**
     - Index of time step to check in prediction. This is the index in the context window we are interested in predicting. Note that time_step_to_check must be within context window, possible values are in [0, context_window -1]. Future implementation will also support multiple indices.
   * - **feature_to_check**
     - Index or indices of features to check in prediction.
   * - **hidden_dim**
     - Hidden layer dimensions (single integer or list for multiple layers).

**Optional parameters**

**Data configuration**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **train_size**
     - Proportion of data to use for training (default: 0.8)
   * - **val_size**
     - Proportion of data to use for validation (default: 0.1)
   * - **test_size**
     - Proportion of data to use for testing (default: 0.1)
   * - **id_columns**
     - Column(s) to process data by groups (default: None)
   * - **feature_names**
     - Custom names for features (default: None)

**Data preprocessing**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **imputer**
     - DataImputer instance for handling missing values (default: None)
   * - **normalize**
     - Whether to normalize the data (default: False)
   * - **normalization_method**
     - Method for normalization (default: "minmax")
       
       - "minmax": Min-Max scaling
       - "zscore": Standard scaling

**Model architecture**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **form**
     - Neural network architecture type (default: "lstm")
       
       - "lstm": Long Short-Term Memory
       - "gru": Gated Recurrent Unit
       - "rnn": Simple RNN
       - "dense": Fully Connected 

.. warning::
    The Dense architecture is available for individual encoders and decoders, but **not implemented for time series autoencoders**. 
    If you select "dense" as the form parameter, the autoencoder will raise an error.

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **bidirectional_encoder**
     - Whether to use bidirectional layers in encoder (default: False)
   * - **bidirectional_decoder**
     - Whether to use bidirectional layers in decoder (default: False)
   * - **activation_encoder**
     - Activation function for encoder layers (default: None). Available options:
       
       - "sigmoid": Sigmoid activation function (outputs between 0 and 1)
       - "tanh": Hyperbolic tangent activation function (outputs between -1 and 1)
       - "relu": Rectified Linear Unit (outputs 0 for negative inputs, linear for positive)
       - "elu": Exponential Linear Unit (smoother than ReLU)
       - "selu": Scaled Exponential Linear Unit (self-normalizing)
       - "softmax": Softmax activation (outputs sum to 1)
       - "softplus": Softplus activation (smooth approximation of ReLU)
       - "softsign": Softsign activation (smooth approximation of tanh)
       - "hard_sigmoid": Hard sigmoid (piecewise linear approximation)
       - "exponential": Exponential activation
       - "linear": Linear activation (no transformation)
       - None: No activation function
   * - **activation_decoder**
     - Activation function for decoder layers (default: None)
       
       Same options as activation_encoder
   * - **use_post_decoder_dense**
     - Whether to add a dense layer after the decoder (default: False)

**Training configuration**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **batch_size**
     - Batch size for training (default: 32)
   * - **epochs**
     - Number of training epochs (default: 100)
   * - **optimizer**
     - Optimizer to use (default: "adam"). Available options:
       
       - "adam": Adaptive Moment Estimation
       - "sgd": Stochastic Gradient Descent
       - "rmsprop": Root Mean Square Propagation
       - "adagrad": Adaptive Gradient Algorithm
       - "adadelta": Adaptive Delta
       - "adamax": Adam with infinity norm
       - "nadam": Nesterov Adam
   * - **use_mask**
     - Whether to use masking for missing values (default: False)
       
       If True and no custom_mask is provided, a mask will be automatically created:
       
       - 0 for null/missing values
       - 1 for non-null values
       
       If True and custom_mask is provided, the provided mask will be used instead
   * - **custom_mask**
     - Custom mask array for missing values. Must match the exact format of the training data:
       
       - If data is a single DataFrame/array: mask should be a numpy array with same shape
       - If data is a tuple of (train, val, test): mask should be a tuple of three arrays with matching shapes
       - If data includes ID columns: mask should preserve the same ID structure
   * - **shuffle**
     - Whether to shuffle the data during training (default: False)
   * - **shuffle_buffer_size**
     - Buffer size for shuffling (default: None, set to dataset size if shuffle=True)

**Early stopping and checkpointing**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **patience**
     - Number of epochs to wait before early stopping (default: 10)
   * - **use_early_stopping**
     - Whether to use early stopping (default: True)
   * - **checkpoint**
     - Save model checkpoint every N epochs (default: 10)

**Logging and visualization**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **verbose**
     - Whether to print detailed information during training (default: False)
   * - **save_path**
     - Directory path to save model checkpoints and plots (default: "autoencoder" in current directory)

**Feature configuration**

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Parameter
     - Description
   * - **feature_names**
     - Custom names for features (default: None)
   * - **feature_weights**
     - Weights for each feature in loss calculation (default: None)
       
       Can be a list of weights with length equal to the number of features. Higher weights will increase the importance of those features in the loss function.

Loss function calculation
~~~~~~~~~~~~~~~~~~~~~~~

The autoencoder uses Mean Squared Error (MSE) as its default loss function, which is calculated as follows:

1. **Basic MSE calculation**:
   - For each time step and feature, the loss is calculated as: MSE = (x - x̂)²
   - Where x is the original value and x̂ is the reconstructed value
   - The final loss is the mean of all squared differences

2. **Feature weighting**:
   - If feature_weights is provided, each feature's contribution to the loss is weighted
   - Weighted MSE = Σ(wᵢ * (xᵢ - x̂ᵢ)²) / Σ(wᵢ)
   - Where wᵢ is the weight for feature i

3. **Masked loss**:
   - When use_mask=True, the loss is only calculated for non-masked positions
   - Masked MSE = Σ(mᵢ * (xᵢ - x̂ᵢ)²) / Σ(mᵢ)
   - Where mᵢ is 1 for non-masked positions and 0 for masked positions

4. **Time step selection**:
   - The loss can be focused on specific time steps using time_step_to_check
   - This is useful when certain time steps are more important for reconstruction

Example of loss calculation with different configurations:

.. code-block:: python

    # Basic MSE without weights or masks
    loss = mean_squared_error(original_data, reconstructed_data)

    # Weighted MSE with feature weights
    loss = weighted_mean_squared_error(
        original_data,
        reconstructed_data,
        feature_weights=[1.0, 2.0, 0.5]  # Higher weight for second feature
    )

    # Masked MSE for handling missing values
    loss = masked_mean_squared_error(
        original_data,
        reconstructed_data,
        mask=mask  # 1 for valid values, 0 for missing values
    )

Input data examples
~~~~~~~~~~~~~~~~

**Basic configuration with automatic splitting**

.. code-block:: python

    # Import required libraries
    import pandas as pd
    import numpy as np
    
    # Create a sample time series DataFrame with 100 time steps and 3 features
    time_steps = 100
    features = 3
    time_series_df = pd.DataFrame(
        np.random.randn(time_steps, features),
        columns=['temperature', 'humidity', 'pressure']
    )
    
    # Initialize and train the autoencoder
    autoencoder = AutoEncoder()
    autoencoder.build_and_train(
        context_window=10,
        data=time_series_df,  # DataFrame with shape (100, 3)
        time_step_to_check=[0],
        feature_to_check=[0, 1],
        hidden_dim=64,
        form="lstm",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )

**Manual data splitting**

.. code-block:: python

    # Import required libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Create a sample time series DataFrame
    time_steps = 100
    features = 3
    time_series_df = pd.DataFrame(
        np.random.randn(time_steps, features),
        columns=['temperature', 'humidity', 'pressure']
    )
    
    # Manually split the data
    train_data, temp_data = train_test_split(time_series_df, train_size=0.8, shuffle=False)
    val_data, test_data = train_test_split(temp_data, train_size=0.5, shuffle=False)
    
    # Initialize and train the autoencoder with pre-split data
    autoencoder = AutoEncoder()
    autoencoder.build_and_train(
        context_window=10,
        data=(train_data, val_data, test_data),  # Tuple of three DataFrames
        time_step_to_check=[0],
        feature_to_check=[0, 1],
        hidden_dim=64,
        form="lstm"
    )

**Custom preprocessing**

.. code-block:: python

    # Create custom imputer
    imputer = DataImputer(strategy="knn", k_neighbors=5)
    
    # Initialize and train the autoencoder with custom preprocessing
    autoencoder = AutoEncoder()
    autoencoder.build_and_train(
        context_window=10,
        data=time_series_df,  # DataFrame with missing values
        time_step_to_check=[0],
        feature_to_check=[0, 1],
        hidden_dim=[128, 64, 32],
        form="lstm",
        imputer=imputer,
        normalize=True,
        normalization_method="minmax",
        bidirectional_encoder=True,
        bidirectional_decoder=True
    )


Usage
~~~~~

The AutoEncoder can be used in two ways:

1. Using the combined `build_and_train` method for a streamlined workflow
2. Using separate `build_model` and `train` methods for more control over the process

**Basic usage with build_and_train**

The simplest way to use the autoencoder is with the combined `build_and_train` method:

.. code-block:: python

    from mango_time_series.models import AutoEncoder
    import pandas as pd
    import numpy as np
    
    # Create a sample time series DataFrame
    time_steps = 100
    features = 3
    time_series_df = pd.DataFrame(
        np.random.randn(time_steps, features),
        columns=['temperature', 'humidity', 'pressure']
    )

    # Initialize the autoencoder
    autoencoder = AutoEncoder()

    # Build and train the model in one step
    autoencoder.build_and_train(
        context_window=10,
        data=time_series_df,  # DataFrame with shape (100, 3)
        time_step_to_check=[0],
        feature_to_check=[0, 1],
        hidden_dim=64,
        form="lstm",
        bidirectional_encoder=True,
        bidirectional_decoder=True,
        normalize=True,
        normalization_method="minmax",
        epochs=100
    )

    # After training, always reconstruct to evaluate the model
    autoencoder.reconstruct()

**Separate build and train**

For more control over the process, you can separate the model building and training steps:

.. code-block:: python

    from mango_time_series.models import AutoEncoder
    import pandas as pd
    import numpy as np
    
    # Create a sample time series DataFrame
    time_steps = 100
    features = 3
    time_series_df = pd.DataFrame(
        np.random.randn(time_steps, features),
        columns=['temperature', 'humidity', 'pressure']
    )

    # Initialize the autoencoder
    autoencoder = AutoEncoder()

    # First, build the model
    autoencoder.build_model(
        context_window=10,
        data=time_series_df,  # DataFrame with shape (100, 3)
        time_step_to_check=[0],
        feature_to_check=[0, 1],
        hidden_dim=64,
        form="lstm",
        bidirectional_encoder=True,
        bidirectional_decoder=True,
        normalize=True,
        normalization_method="minmax"
    )

    # Then train the model with specific training parameters
    autoencoder.train(
        epochs=100,
        batch_size=32,
        checkpoint=10,
        use_early_stopping=True,
        patience=10
    )

    # After training, always reconstruct to evaluate the model
    autoencoder.reconstruct()

**Evaluating the model with reconstruct**

The `reconstruct` method generates several visualizations to evaluate the model's performance on the training data:

1. **Reconstruction Plot**: Shows the actual vs. reconstructed data for each feature
   - Uses `plot_actual_and_reconstructed` from `mango_time_series.models.utils.plots`
   - Displays time series data with actual values in blue and reconstructed values in red
   - Includes feature names and time step information
   - Saved as "reconstruction.png" in the specified save_path

2. **Loss History Plot**: Shows the training and validation loss over epochs
   - Uses `plot_loss_history` from `mango_time_series.models.utils.plots`
   - Displays training loss in blue and validation loss in red
   - Includes epoch information and loss values
   - Saved as "loss_history.png" in the specified save_path

**Using the trained model with reconstruct_new_data**

Once you have trained and evaluated your model using `reconstruct`, you can use the `reconstruct_new_data` method to apply the trained autoencoder to new, unseen data. 

The method supports iterative reconstruction in case of missing values, where the model can refine its output over multiple passes, potentially improving the quality of the reconstruction.

Example of reconstruct_new_data usage:

.. code-block:: python

    # Reconstruct new data with multiple iterations
    results = autoencoder.reconstruct_new_data(
        new_data,
        iterations=3,  # Number of reconstruction iterations
        id_columns=["id"],  # Columns to identify different time series
        save_path="path/to/save"  # Where to save the results and plots
    )
    

**Visualizations for reconstruct_new_data**

The `reconstruct_new_data` method generates several visualizations to help analyze the reconstruction of new data:

1. **Reconstruction Plot**: Similar to the one in `reconstruct`, but for the new data
   - Uses `plot_actual_and_reconstructed` from `mango_time_series.models.utils.plots`
   - Shows actual vs. reconstructed values for each feature
   - Includes feature names and time step information
   - Saved as "reconstruction_new_data.png" in the specified save_path

2. **Reconstruction Iterations Plot**: Shows how the reconstruction improves over iterations
   - Uses `plot_reconstruction_iterations` from `mango_time_series.models.utils.plots`
   - Displays the evolution of reconstructed values across iterations
   - Includes feature names and iteration information
   - Saved as "reconstruction_iterations.png" in the specified save_path

3. **Error Distribution Plot**: Shows the distribution of reconstruction errors
   - Uses `plot_error_distribution` from `mango_time_series.models.utils.plots`
   - Displays histograms of reconstruction errors for each feature
   - Includes feature names and error statistics
   - Saved as "error_distribution.png" in the specified save_path

Model persistence
~~~~~~~~~~~~~~~

During training, the model is automatically saved in two ways:

1. **Checkpoints**: Every N epochs (specified by the `checkpoint` parameter, default: 10)
2. **Best model**: The model with the best validation loss is saved at the end of training

In addition to model weights, the persistence mechanism now stores all necessary metadata for future reconstruction and inference. This includes:

- Normalization parameters (min-max values or z-score statistics)
- Feature names and order
- Time steps and features used for reconstruction
- ID-based normalization structure (if applicable)
- The normalization method used during training

This ensures that when a model is loaded for inference, it applies the same preprocessing steps as during training, avoiding inconsistencies or the need to reconfigure the environment.

You can also manually save and load models using the following methods:

.. code-block:: python

    # Manually save model (useful for saving intermediate states)
    autoencoder.save(save_path="models", filename="my_model.pkl")
    
    # Load a previously saved model
    loaded_model = AutoEncoder.load_from_pickle("models/my_model.pkl")
    
    # Use the loaded model to reconstruct new data
    results = loaded_model.reconstruct_new_data(new_data)

Once loaded, the model can reconstruct new data without requiring re-specification of preprocessing settings, as all relevant parameters are embedded in the saved object.