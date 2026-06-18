import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_analysis_utils import AnalyzeDataset


class TargetVisualizationAutoFitTests(unittest.TestCase):
    @staticmethod
    def _analyzer(normal=True, uniform=True):
        return AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=[
                ("chi2", 0.05, None),
            ],
            good_of_fit_uniform_test_instructions=uniform,
            normal_test_instructions=normal,
            check_assumptions=False,
        )

    @staticmethod
    def _seed_target(analyzer, target, target_type):
        metadata = analyzer._blank_target_dict()
        metadata["target_dtype"] = [target_type]
        analyzer.target_key_feature_meta_vals[target] = metadata
        analyzer.has_called_fit_column_relationships.add(target)
        return metadata

    @staticmethod
    def _fit_visualizations(analyzer, data, target, auto_fit=True):
        return analyzer._fit_target_visualizations(
            data,
            targets=target,
            reject_numcat=False,
            reject_numnum=True,
            reject_catcat=False,
            is_super_or_subcat=False,
            not_uniform_or_reject_normal=True,
            reject_multivariates=False,
            auto_fit=auto_fit,
            check_assumptions=False,
        )

    def test_numeric_metadata_routes_integer_float_and_nullable_targets(self):
        target_series = (
            pd.Series(range(20), dtype="int64"),
            pd.Series(np.linspace(0, 1, 20), dtype="float32"),
            pd.Series(range(20), dtype="Int64"),
        )

        for series in target_series:
            with self.subTest(dtype=series.dtype):
                data = pd.DataFrame({"target": series})
                analyzer = self._analyzer()
                self._seed_target(analyzer, "target", "numeric")

                def record_fit(df, numeric_columns=None):
                    analyzer.has_called_fit_normal.add(numeric_columns)
                    return analyzer

                with patch.object(
                    analyzer,
                    "fit_normal",
                    side_effect=record_fit,
                ) as fit_normal:
                    self._fit_visualizations(analyzer, data, "target")

                fit_normal.assert_called_once_with(
                    data,
                    numeric_columns="target",
                )
                self.assertIn("target", analyzer.has_called_fit_normal)

    def test_enabled_normality_autofit_updates_numeric_target(self):
        data = pd.DataFrame(
            {
                "target": np.linspace(0, 1, 20),
            }
        )
        analyzer = self._analyzer()
        self._seed_target(analyzer, "target", "numeric")

        self._fit_visualizations(analyzer, data, "target")

        self.assertIn("target", analyzer.has_called_fit_normal)
        self.assertTrue(
            analyzer.target_key_feature_meta_vals["target"][
                "is_normal_or_uniform"
            ]
        )

    def test_enabled_uniform_autofit_updates_categorical_target(self):
        data = pd.DataFrame(
            {
                "target": ["A"] * 20 + ["B"] * 20,
            }
        )
        analyzer = self._analyzer()
        self._seed_target(analyzer, "target", "categoric")

        self._fit_visualizations(analyzer, data, "target")

        self.assertIn(
            "target",
            analyzer.has_called_fit_goodness_of_fit_uniform,
        )
        self.assertTrue(
            analyzer.target_key_feature_meta_vals["target"][
                "is_normal_or_uniform"
            ]
        )

    def test_disabled_normality_warns_and_preserves_other_plot_data(self):
        data = pd.DataFrame(
            {
                "target": np.linspace(0, 1, 20),
                "feature": np.linspace(0, 2, 20),
            }
        )
        analyzer = self._analyzer(normal=False)
        metadata = self._seed_target(analyzer, "target", "numeric")
        metadata["significant_numeric_relationships"] = ["feature"]
        metadata["significant_numeric_tests"] = ["pearson"]

        with self.assertWarnsRegex(
            UserWarning,
            "Normality testing is disabled",
        ):
            results = self._fit_visualizations(
                analyzer,
                data,
                "target",
            )

        self.assertEqual(
            results["target"]["reject_numnum"],
            [["target", "feature", "pearson"]],
        )
        self.assertNotIn("target", analyzer.has_called_fit_normal)

    def test_disabled_uniform_test_warns_and_continues(self):
        data = pd.DataFrame({"target": ["A", "B"] * 20})
        analyzer = self._analyzer(uniform=False)
        self._seed_target(analyzer, "target", "categoric")

        with self.assertWarnsRegex(
            UserWarning,
            "Uniform goodness-of-fit testing is disabled",
        ):
            results = self._fit_visualizations(
                analyzer,
                data,
                "target",
            )

        self.assertIn("target", results)
        self.assertNotIn(
            "target",
            analyzer.has_called_fit_goodness_of_fit_uniform,
        )

    def test_already_fitted_target_is_not_fitted_again(self):
        data = pd.DataFrame({"target": np.linspace(0, 1, 20)})
        analyzer = self._analyzer()
        self._seed_target(analyzer, "target", "numeric")
        analyzer.has_called_fit_normal.add("target")

        with patch.object(analyzer, "fit_normal") as fit_normal:
            self._fit_visualizations(analyzer, data, "target")

        fit_normal.assert_not_called()

    def test_auto_fit_false_retains_explicit_unfitted_error(self):
        data = pd.DataFrame({"target": np.linspace(0, 1, 20)})
        analyzer = self._analyzer()
        self._seed_target(analyzer, "target", "numeric")

        with self.assertRaisesRegex(
            RuntimeError,
            r"fit_normal\(\) needs to be called",
        ):
            self._fit_visualizations(
                analyzer,
                data,
                "target",
                auto_fit=False,
            )


if __name__ == "__main__":
    unittest.main()
