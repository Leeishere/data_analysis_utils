import unittest

import numpy as np
import pandas as pd
import scipy.stats

from data_analysis_utils.ANOVA import ANOVA


class TwoSampleTTestsTests(unittest.TestCase):
    def setUp(self):
        self.anova = ANOVA()

    def test_valid_groups_return_welch_comparison(self):
        data = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = self.anova.two_sample_t_tests(data)
        expected_p_value = scipy.stats.ttest_ind(
            [1.0, 2.0],
            [3.0, 4.0],
            equal_var=False,
        ).pvalue

        self.assertEqual(result[["subcat_1", "subcat_2"]].values.tolist(), [["a", "b"]])
        self.assertAlmostEqual(result.loc[0, "P-value"], expected_p_value)
        self.assertEqual(result.loc[0, "n_samples_1"], 2)
        self.assertEqual(result.loc[0, "n_samples_2"], 2)
        self.assertEqual(result["n_samples_1"].dtype, np.dtype("int64"))
        self.assertEqual(result["n_samples_2"].dtype, np.dtype("int64"))

    def test_invalid_groups_are_omitted_by_default(self):
        data = pd.DataFrame(
            {
                "group": ["a", "b", "b", "c", "c"],
                "value": [1.0, 2.0, 3.0, 4.0, 6.0],
            }
        )

        result = self.anova.two_sample_t_tests(data)

        self.assertEqual(result[["subcat_1", "subcat_2"]].values.tolist(), [["b", "c"]])
        self.assertTrue(result["P-value"].notna().all())

    def test_include_invalid_retains_pairs_with_nan_p_values(self):
        data = pd.DataFrame(
            {
                "group": ["a", "b", "b", "c", "c"],
                "value": [1.0, 2.0, 3.0, 4.0, 6.0],
            }
        )

        result = self.anova.two_sample_t_tests(data, include_invalid=True)

        self.assertEqual(
            result[["subcat_1", "subcat_2"]].values.tolist(),
            [["a", "b"], ["a", "c"], ["b", "c"]],
        )
        self.assertTrue(result.loc[:1, "P-value"].isna().all())
        self.assertTrue(np.isfinite(result.loc[2, "P-value"]))
        self.assertEqual(result["n_samples_1"].dtype, pd.Int64Dtype())
        self.assertEqual(result["n_samples_2"].dtype, pd.Int64Dtype())

    def test_zero_variance_group_is_invalid(self):
        data = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b"],
                "value": [1.0, 1.0, 2.0, 3.0],
            }
        )

        default_result = self.anova.two_sample_t_tests(data)
        diagnostic_result = self.anova.two_sample_t_tests(data, include_invalid=True)

        self.assertTrue(default_result.empty)
        self.assertEqual(diagnostic_result.shape[0], 1)
        self.assertTrue(pd.isna(diagnostic_result.loc[0, "P-value"]))

    def test_fewer_than_two_groups_returns_empty_result(self):
        cases = [
            pd.DataFrame({"group": pd.Series(dtype=object), "value": pd.Series(dtype=float)}),
            pd.DataFrame({"group": ["a", "a"], "value": [1.0, 2.0]}),
        ]

        for data in cases:
            with self.subTest(number_of_rows=data.shape[0]):
                result = self.anova.two_sample_t_tests(
                    data,
                    include_invalid=True,
                )

                self.assertTrue(result.empty)
                self.assertEqual(
                    result.columns.tolist(),
                    [
                        "subcat_1",
                        "subcat_2",
                        "P-value",
                        "n_samples_1",
                        "n_samples_2",
                    ],
                )
                self.assertEqual(result["n_samples_1"].dtype, pd.Int64Dtype())
                self.assertEqual(result["n_samples_2"].dtype, pd.Int64Dtype())

    def test_sample_counts_exclude_missing_numeric_values(self):
        data = pd.DataFrame(
            {
                "group": ["a", "a", "a", "b", "b"],
                "value": [1.0, np.nan, 2.0, 3.0, 4.0],
            }
        )

        result = self.anova.two_sample_t_tests(data)

        self.assertEqual(result.loc[0, "n_samples_1"], 2)
        self.assertEqual(result.loc[0, "n_samples_2"], 2)
        self.assertTrue(np.isfinite(result.loc[0, "P-value"]))

    def test_subcategory_similarities_can_expose_invalid_comparisons(self):
        data = pd.DataFrame(
            {
                "group": ["a", "b", "b"],
                "value": [1.0, 2.0, 3.0],
            }
        )

        default_result = self.anova.subcategory_similarities(
            data,
            return_similar=True,
        )
        diagnostic_result = self.anova.subcategory_similarities(
            data,
            return_similar=True,
            include_invalid=True,
        )

        self.assertTrue(default_result.empty)
        self.assertEqual(diagnostic_result.shape[0], 1)
        self.assertTrue(pd.isna(diagnostic_result.loc[0, "P-value"]))


if __name__ == "__main__":
    unittest.main()
