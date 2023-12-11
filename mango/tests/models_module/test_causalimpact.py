import unittest

import numpy as np
import pandas as pd
import tensorflow_probability
from causalimpact import Seasons, ModelOptions

from mango.models.causal_impact import CausalImpactSTSModel


class TestCausalImpactSTSModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(np.random.rand(100, 1), columns=["y"])
        self.data_with_covariates = pd.DataFrame(
            np.random.rand(100, 2), columns=["y", "x"]
        )
        self.pre_period = [0, 49]
        self.post_period = [50, 99]

    def test_run_with_valid_data(self):
        # Without covariates
        model = CausalImpactSTSModel(self.data, self.pre_period, self.post_period)
        model.run()
        self.assertIsNotNone(model.fitted_model)

        # With covariates
        model = CausalImpactSTSModel(
            self.data_with_covariates, self.pre_period, self.post_period
        )
        model.run()
        self.assertIsNotNone(model.fitted_model)

    def test_run_with_invalid_pre_period(self):
        with self.assertRaises(IndexError):
            model = CausalImpactSTSModel(self.data, [0, 101], self.post_period)
            model.run()

    def test_run_with_invalid_post_period(self):
        with self.assertRaises(IndexError):
            model = CausalImpactSTSModel(self.data, self.pre_period, [50, 101])
            model.run()

    def test_run_with_overlapping_periods(self):
        with self.assertRaises(ValueError):
            model = CausalImpactSTSModel(self.data, [0, 60], [50, 99])
            model.run()

    def test_parse_seasons_options_with_valid_input(self):
        model = CausalImpactSTSModel(self.data, self.pre_period, self.post_period)
        seasons_options = [Seasons(num_seasons=2, num_steps_per_season=1)]
        result = model._parse_seasons_options(seasons_options)
        self.assertEqual(result, seasons_options)

        seasons_options = Seasons(num_seasons=1, num_steps_per_season=1)
        result = model._parse_seasons_options(seasons_options)
        self.assertEqual(result, [seasons_options])

    def test_parse_seasons_options_with_invalid_input(self):
        model = CausalImpactSTSModel(self.data, self.pre_period, self.post_period)
        seasons_options = "invalid_input"
        with self.assertRaises(ValueError):
            model._parse_seasons_options(seasons_options)

    def test_options_in_init(self):
        seasons_options = Seasons(num_seasons=2, num_steps_per_season=1)
        model = CausalImpactSTSModel(
            self.data,
            self.pre_period,
            self.post_period,
            seasons_options=seasons_options,
            prior_level_sd=1,
        )
        self.assertEqual(model.seasons_options, [seasons_options])
        self.assertEqual(
            model.model_options,
            ModelOptions(seasons=[seasons_options], prior_level_sd=1),
        )

    def test_run_with_linear_trend(self):
        model = CausalImpactSTSModel(
            self.data,
            self.pre_period,
            self.post_period,
            add_linear_trend_experimental=True,
            seasons_options=Seasons(num_seasons=2, num_steps_per_season=1),
            prior_level_sd=1,
        )
        # Assert model.custom_model is not None
        self.assertIsNotNone(model.custom_model)
        # Assert model.custom_model is a STS.Sum model
        self.assertIsInstance(
            model.custom_model,
            tensorflow_probability.sts.Sum,
        )
        # Assert model.fitted_model.components is a list of length 2
        self.assertEqual(len(model.custom_model.components), 2)
        # Assert there is a linear trend component in the model and a seasonal component
        self.assertIsInstance(
            model.custom_model.components[0],
            tensorflow_probability.sts.LocalLinearTrend,
        )
        self.assertIsInstance(
            model.custom_model.components[1],
            tensorflow_probability.sts.Seasonal,
        )
        model.run()
        self.assertIsNotNone(model.fitted_model)

        # With covariates
        model = CausalImpactSTSModel(
            self.data_with_covariates,
            self.pre_period,
            self.post_period,
            add_linear_trend_experimental=True,
            seasons_options=Seasons(num_seasons=2, num_steps_per_season=1),
            prior_level_sd=1,
        )
        # Assert model.custom_model is not None
        self.assertIsNotNone(model.custom_model)
        # Assert model.custom_model is a STS.Sum model
        self.assertIsInstance(
            model.custom_model,
            tensorflow_probability.sts.Sum,
        )
        # Assert model.fitted_model.components is a list of length 3
        self.assertEqual(len(model.custom_model.components), 3)
        # Assert there is a linear trend component in the model and a seasonal component
        self.assertIsInstance(
            model.custom_model.components[0],
            tensorflow_probability.sts.LocalLinearTrend,
        )
        self.assertIsInstance(
            model.custom_model.components[1],
            tensorflow_probability.python.experimental.sts_gibbs.gibbs_sampler.SpikeAndSlabSparseLinearRegression,
        )
        self.assertIsInstance(
            model.custom_model.components[2],
            tensorflow_probability.sts.Seasonal,
        )
        model.run()
        self.assertIsNotNone(model.fitted_model)

    def test_summary(self):
        model = CausalImpactSTSModel(
            self.data,
            self.pre_period,
            self.post_period,
        )
        model.run()
        figure, summary, predictions = model.summarize_results()
        # Assertion tests
        self.assertIsNotNone(figure)
        self.assertIsNotNone(summary)
        # self.assertIsNotNone(predictions)
        # Run in "es" language
        figure, summary, predictions = model.summarize_results(lang="es")
        # Assert figure is in Spanish

        # Assert columns and index of summary are in Spanish
        self.assertListEqual(
            summary.columns.to_list(),
            [
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
            ],
        )
        self.assertListEqual(
            summary.index.to_list(),
            [
                "Promedio",
                "Acumulado",
            ],
        )

        # Invalid language
        with self.assertRaises(ValueError):
            figure, summary, predictions = model.summarize_results(lang="invalid")
