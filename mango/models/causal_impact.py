import logging
import os.path
import pathlib
import pkg_resources
import tomllib as toml
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Literal

import causalimpact.data as cid
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from causalimpact import (
    fit_causalimpact,
    ModelOptions,
    InferenceOptions,
    Seasons,
    DataOptions,
)
from causalimpact.plot import plot as plot_ci
from tensorflow_probability.python.experimental.distributions import (
    MultivariateNormalPrecisionFactorLinearOperator,
)
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions
TensorLike = tf.types.experimental.TensorLike


class BaseCausalImpact(ABC):
    def __init__(self, data, pre_period, post_period):
        self.data = data
        self.pre_period = pre_period
        self.post_period = post_period

    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses should implement this!")


class CausalImpactSTSModel(BaseCausalImpact):
    """
    A class used to represent the Causal Impact Structural Time Series Model.

    Attributes
    ----------
    data : pandas.DataFrame
        a DataFrame containing the data to be used in the model
    pre_period : tuple
        a tuple representing the pre-intervention period
    post_period : tuple
        a tuple representing the post-intervention period
    model_options : dict, optional
        a dictionary of options for the model (default is None)
    inference_options : dict, optional
        a dictionary of options for inference (default is None)
    custom_model : tfp.sts.StructuralTimeSeries, optional
        a custom model to be used instead of the default (default is None)

    Methods
    -------
    run():
        Builds and fits the model.
    summarize_results(plot_kwargs=None, lang='en'):
        Summarizes the results of the fitted model, including generating plots and returning a summary and predictions.
    create_dashboard():
        Creates mango dashboard in case dashboards dependencies are installed.
    """

    def __init__(
        self,
        data,
        pre_period,
        post_period,
        prior_level_sd: float = 0.01,
        outcome_column: str = None,
        standardize_data: bool = True,
        dtype: tf.dtypes.DType = tf.float32,
        seasons_options: Union[List[Dict[str, int]], Seasons, List[Seasons]] = None,
        inference_options: Union[Dict[str, int], InferenceOptions] = None,
        add_linear_trend_experimental: bool = False,
    ):
        """
        Constructs all the necessary attributes for the CausalImpactSTSModel object.

        :param data: a pandas DataFrame containing the data to be used in the model. Index is assumed to be time.
        :param pre_period: a tuple representing the pre-intervention
        :param post_period: a tuple representing the post-intervention
        :param prior_level_sd: prior level standard deviation for the local level component of the model
        :param outcome_column: name of the outcome column in the data. If None, first column is assumed to be the
        outcome variable.
        :param standardize_data: whether to standardize the data or not
        :param dtype: data type to handle the data
        :param seasons_options: a list of Seasons objects or a Seasons object. If None, no seasonal component is added.
        :param inference_options: a dictionary of options for inference. If None, default options are used.
        :param add_linear_trend_experimental: whether to add a linear trend component to the model or not. This is
        experimental and may not work as expected.
        """
        # Initialize parent class
        super().__init__(data, pre_period, post_period)

        # Set up attributes
        self.custom_model = None
        self.fitted_model = None

        # Show warning if outcome column is None
        if outcome_column is None:
            logging.warning(f"Assuming first column '{data.columns[0]}' is outcome.")

        # Parse options
        self.seasons_options = self._parse_seasons_options(seasons_options)
        self.inference_options = self._parse_inference_options(inference_options)
        self.data_options = DataOptions(
            outcome_column=outcome_column,
            standardize_data=standardize_data,
            dtype=dtype,
        )
        if self.seasons_options:
            self.model_options = ModelOptions(
                seasons=self.seasons_options, prior_level_sd=prior_level_sd
            )
        else:
            self.model_options = ModelOptions()

        # Build custom model if add_linear_trend_experimental is True
        if add_linear_trend_experimental:
            # Original library does not support this. This is experimental. There may be a good reason why this is not
            # supported in the original library.
            logging.warning("Adding linear trend. This is experimental.")
            logging.warning(
                "Consider passing as a covariable instead of adding linear trend as follows:\n"
                "data['trend'] = [i for i in range(len(data))]\n"
                "Other trends can be handled similarly and give better results.\n"
                "The better results are found using co-variates that follow same trends.\n"
            )
            self.custom_model = self._build_structural_time_series_model_with_trend()

    def create_dashboard(self):
        """Creates mango dashboard in case dashboards dependency is installed."""
        pass

    def _parse_seasons_options(
        self, seasons_options: Union[Seasons, List[Seasons]]
    ) -> Optional[List[Seasons]]:
        if isinstance(seasons_options, list) and all(
            isinstance(season, Seasons) for season in seasons_options
        ):
            return seasons_options
        elif isinstance(seasons_options, Seasons):
            return [seasons_options]
        elif seasons_options is None:
            return None
        else:
            raise ValueError(
                "Seasons options should be a list of dictionaries or a Seasons object or None."
            )

    def _parse_inference_options(
        self, inference_options: Dict[str, int]
    ) -> InferenceOptions:
        return InferenceOptions(
            num_results=inference_options.get("inference_options", 900),
            num_warmup_steps=inference_options.get("num_warmup_steps", 100),
        )

    def _build_structural_time_series_model_with_trend(self):
        """Build custom model with linear trend."""

        # Prepare data
        ci_data = cid.CausalImpactData(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            outcome_column=self.data_options.outcome_column,
            standardize_data=self.data_options.standardize_data,
            dtype=self.data_options.dtype,
        )

        # Prepare outcome data
        after_pre_period_length = ci_data.model_after_pre_data.shape[0]
        extended_outcome_ts = tfp.sts.MaskedTimeSeries(
            time_series=tf.concat(
                [
                    ci_data.outcome_ts.time_series,
                    tf.fill(
                        after_pre_period_length,
                        tf.constant(float("nan"), dtype=tf.float32),
                    ),
                ],
                axis=0,
            ),
            is_missing=tf.concat(
                [ci_data.outcome_ts.is_missing, tf.fill(after_pre_period_length, True)],
                axis=0,
            ),
        )
        outcome_sd = tf.convert_to_tensor(
            np.nanstd(ci_data.outcome_ts.time_series, ddof=1), dtype=tf.float32
        )

        # Setup seasonal components
        seasonal_components = []
        seasonal_variance_prior = tfd.InverseGamma(
            concentration=0.005, scale=5e-7 * tf.square(outcome_sd)
        )
        seasonal_variance_prior.upper_bound = outcome_sd
        if self.seasons_options is not None:
            for seasonal_options in self.seasons_options:
                seasonal_components.append(
                    tfp.sts.Seasonal(
                        num_seasons=seasonal_options.num_seasons,
                        num_steps_per_season=np.array(
                            seasonal_options.num_steps_per_season
                        ),
                        allow_drift=True,
                        constrain_mean_effect_to_zero=True,
                        drift_scale_prior=tfd.TransformedDistribution(
                            bijector=tfb.Invert(tfb.Square()),
                            distribution=seasonal_variance_prior,
                        ),
                        initial_effect_prior=tfd.Normal(loc=0.0, scale=outcome_sd),
                    )
                )

        # Setup design matrix
        if ci_data.feature_ts is not None:
            logging.warning(
                "There are covariables, please consider setting add_linear_trend_experimental to False."
            )
            design_matrix = ci_data.feature_ts.values
            design_matrix = tf.convert_to_tensor(design_matrix, dtype=tf.float32)
        else:
            design_matrix = None

        # Weights prior (just guessing this is the Spike and Slab prior)
        if design_matrix is not None:
            design_shape = ps.shape(design_matrix)
            num_outputs = design_shape[-2]
            num_dimensions = design_shape[-1]
            sparse_weights_nonzero_prob = tf.minimum(
                tf.constant(1.0, dtype=tf.float32), 3.0 / num_dimensions
            )
            x_transpose_x = tf.matmul(design_matrix, design_matrix, transpose_a=True)
            weights_prior_precision = (
                0.01
                * tf.linalg.set_diag(
                    0.5 * x_transpose_x, tf.linalg.diag_part(x_transpose_x)
                )
                / num_outputs
            )
            # TODO(colcarroll): Remove this cholesky - it is used to instantiate the
            # MVNPFLO below, but later code only uses the precision.
            precision_factor = tf.linalg.cholesky(weights_prior_precision)

            # Note that this prior uses the entire design matrix -- not just the
            # pre-period -- which "cheats" by using future data.
            weights_prior = MultivariateNormalPrecisionFactorLinearOperator(
                precision_factor=tf.linalg.LinearOperatorFullMatrix(precision_factor),
                precision=tf.linalg.LinearOperatorFullMatrix(weights_prior_precision),
            )
        else:
            weights_prior = None
            sparse_weights_nonzero_prob = None

        # Local level prior
        prior_level_sd = tf.constant(
            self.model_options.prior_level_sd, dtype=tf.float32
        )
        level_scale = tf.ones([], dtype=tf.float32) * prior_level_sd * outcome_sd
        local_level_prior_sample_size = tf.constant(32.0, dtype=tf.float32)
        level_concentration = tf.cast(
            local_level_prior_sample_size / 2.0, dtype=tf.float32
        )
        level_variance_prior_scale = (
            level_scale * level_scale * (local_level_prior_sample_size / 2.0)
        )
        level_variance_prior = tfd.InverseGamma(
            concentration=level_concentration, scale=level_variance_prior_scale
        )
        level_variance_prior.upper_bound = outcome_sd
        initial_level_prior = tfd.Normal(
            loc=tf.cast(extended_outcome_ts.time_series[..., 0], dtype=tf.float32),
            scale=outcome_sd,
        )

        # Observation noise variance prior
        if design_matrix is not None:
            observation_noise_variance_prior = tfd.InverseGamma(
                concentration=tf.constant(25.0, dtype=tf.float32),
                scale=tf.math.square(outcome_sd) * tf.constant(5.0, dtype=tf.float32),
            )
        else:
            observation_noise_variance_prior = tfd.InverseGamma(
                concentration=tf.constant(0.005, dtype=tf.float32),
                scale=tf.math.square(outcome_sd) * tf.constant(0.005, dtype=tf.float32),
            )
        observation_noise_variance_prior.upper_bound = outcome_sd * tf.constant(
            1.2, dtype=tf.float32
        )

        # Trend variance prior as inverse gamma same as level variance
        trend_variance_prior = level_variance_prior

        # Build model
        return gibbs_sampler.build_model_for_gibbs_fitting(
            extended_outcome_ts,
            design_matrix=design_matrix,
            weights_prior=weights_prior,
            level_variance_prior=level_variance_prior,
            slope_variance_prior=trend_variance_prior,
            observation_noise_variance_prior=observation_noise_variance_prior,
            initial_level_prior=initial_level_prior,
            sparse_weights_nonzero_prob=sparse_weights_nonzero_prob,
            seasonal_components=seasonal_components,
        )

    def run(self):
        """Build model."""
        self.fitted_model = fit_causalimpact(
            data=self.data,
            pre_period=self.pre_period,
            post_period=self.post_period,
            model_options=self.model_options,
            inference_options=self.inference_options,
            seed=33,
            experimental_model=self.custom_model,
        )

    def summarize_results(self, plot_kwargs=None, lang: Literal["en", "es"] = "en"):
        """
        Summarize results of the fitted model.
        :param plot_kwargs: kwargs to pass to plot_ci from tfp-causalimpact library.
        :param lang: language of the summary and figure. Currently only "en" and "es" are supported.
        :return: figure, summary and predictions.
        """
        # Assert supported languages
        if lang not in ["en", "es"]:
            raise ValueError(f"Language '{lang}' is not supported.")
        # Set default plot kwargs
        if plot_kwargs is None:
            plot_kwargs = dict()
        # Generate plots
        fig = plot_ci(self.fitted_model, backend="matplotlib", **plot_kwargs)
        if not isinstance(fig, plt.Figure):
            # This should prevent the old tfp-causalimpact library from doing altair plots which may lead to errors
            logging.warning(
                f"Could not generate figure in matplotlib. This is most probably due to old version of "
                f"tfp-causalimpact library. Must be installed as follows:\n"
                "pip install tfp-causalimpact @ git+https://github.com/google/tfp-causalimpact"
            )
            fig = None
        # If "es" is passed, translate to spanish
        if lang == "es":
            # Change shared x label
            fig.axes[2].set_xlabel("Tiempo")

            # Changed y labels
            fig.axes[0].set_ylabel("Datos")
            fig.axes[1].set_ylabel("Puntual")
            fig.axes[2].set_ylabel("Acumulado")

            # Get legends
            fig_0_legend = fig.axes[0].legend()
            fig_1_legend = fig.axes[1].legend()
            fig_2_legend = fig.axes[2].legend()

            # Change legend texts
            fig_0_legend.get_texts()[0].set_text("Promedio")
            fig_0_legend.get_texts()[1].set_text("Observado")
            fig_1_legend.get_texts()[0].set_text("Efecto puntual")
            fig_2_legend.get_texts()[0].set_text("Efecto acumulado")

            # Set position to upper-left
            fig_0_legend._loc = 2
            fig_1_legend._loc = 2
            fig_2_legend._loc = 2

        # Generate dataframe with summary
        summary = self.fitted_model.summary
        if lang == "es":
            # Modify summary
            summary.columns = [
                "real",
                "predicción",
                "predicción_inferior",
                "predicción_superior",
                "predicción_desviación",
                "efecto_absoluto",
                "efecto_absoluto_inferior",
                "efecto_absoluto_superior",
                "efecto_absoluto_desviación",
                "efecto_relativo",
                "efecto_relativo_inferior",
                "efecto_relativo_superior",
                "efecto_relativo_desviación",
                "p-valor",
                "alfa",
            ]
            summary.index = [
                "Promedio",
                "Acumulado",
            ]

        # Get predictions and actuals with intervals
        # predictions = self.fitted_model
        predictions = None

        return fig, summary, predictions
