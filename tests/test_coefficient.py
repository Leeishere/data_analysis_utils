import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from data_analysis_utils.Coefficient import Coefficient


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.coefficient = Coefficient()
        self.data = pd.DataFrame(
            {
                "target": [1, 2, 3, 4, 5],
                "feature": [1, 4, 9, 16, 25],
                "feature_b": [25, 16, 9, 4, 1],
            }
        )

    def test_target_mode_uses_requested_correlation_method(self):
        pearson = self.data["target"].corr(
            self.data["feature"],
            method="pearson",
        )

        for method in ("pearson", "spearman", "kendall"):
            with self.subTest(method=method):
                result = self.coefficient.test_all_num_num_coefficient(
                    self.data,
                    corr_method=method,
                    numeric_columns="feature",
                    target="target",
                )
                expected = self.data["target"].corr(
                    self.data["feature"],
                    method=method,
                )

                self.assertAlmostEqual(result.loc[0, "Correlation"], expected)

                if method != "pearson":
                    self.assertNotAlmostEqual(
                        result.loc[0, "Correlation"],
                        pearson,
                    )

    def test_target_mode_accepts_a_list_of_targets(self):
        data = self.data.assign(target_b=[5, 4, 3, 2, 1])
        result = self.coefficient.test_all_num_num_coefficient(
            data,
            corr_method="spearman",
            numeric_columns=["feature"],
            target=["target", "target_b"],
        )

        feature_results = result.loc[result["numeric_2"] == "feature"]
        self.assertEqual(
            feature_results["numeric_1"].tolist(),
            ["target", "target_b"],
        )
        self.assertAlmostEqual(
            feature_results.iloc[0]["Correlation"],
            data["target"].corr(data["feature"], method="spearman"),
        )
        self.assertAlmostEqual(
            feature_results.iloc[1]["Correlation"],
            data["target_b"].corr(data["feature"], method="spearman"),
        )

    def test_target_mode_preserves_numeric_column_order(self):
        result = self.coefficient.test_all_num_num_coefficient(
            self.data,
            corr_method="spearman",
            numeric_columns=["feature_b", "feature"],
            target="target",
        )

        self.assertEqual(
            result["numeric_2"].tolist(),
            ["feature_b", "feature"],
        )

    def test_invalid_correlation_method_fails_clearly(self):
        for target in (None, "target"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    ValueError,
                    "Correlation method must be one of",
                ):
                    self.coefficient.test_all_num_num_coefficient(
                        self.data,
                        corr_method="not-a-method",
                        numeric_columns=["target", "feature"],
                        target=target,
                    )

    def test_non_target_mode_is_unchanged(self):
        result = self.coefficient.test_all_num_num_coefficient(
            self.data,
            corr_method="spearman",
            numeric_columns=["target", "feature"],
        )
        expected = pd.DataFrame(
            {
                "numeric_1": ["target"],
                "numeric_2": ["feature"],
                "Correlation": [
                    self.data["target"].corr(
                        self.data["feature"],
                        method="spearman",
                    )
                ],
            }
        )

        assert_frame_equal(result, expected)

    def test_comparison_filters_by_absolute_correlation_strength(self):
        above_threshold = (
            self.coefficient.num_num_column_coefficient_comparison(
                self.data,
                corr_threshold=0.6,
                keep_above_corr=True,
                numeric_columns=["target", "feature_b"],
                corr_method="spearman",
            )
        )
        below_threshold = (
            self.coefficient.num_num_column_coefficient_comparison(
                self.data,
                corr_threshold=0.6,
                keep_above_corr=False,
                numeric_columns=["target", "feature_b"],
                corr_method="spearman",
            )
        )

        self.assertAlmostEqual(above_threshold.loc[0, "Correlation"], -1.0)
        self.assertTrue(below_threshold.empty)


if __name__ == "__main__":
    unittest.main()
