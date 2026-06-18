import unittest

import pandas as pd

from data_analysis_utils import AnalyzeDataset


class AnalyzeDatasetCorrelationClassificationTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("spearman", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

    def test_correlation_classification_uses_absolute_strength(self):
        correlations = [
            ("positive_strong", 0.8),
            ("negative_strong", -0.8),
            ("positive_boundary", 0.6),
            ("negative_boundary", -0.6),
            ("positive_weak", 0.2),
            ("negative_weak", -0.2),
        ]

        for method in ("pearson", "spearman", "kendall"):
            with self.subTest(method=method):
                test_df = pd.DataFrame(
                    {
                        "column_a": ["target"] * len(correlations),
                        "column_b": [name for name, _ in correlations],
                        "test": [method] * len(correlations),
                        "Correlation": [
                            correlation for _, correlation in correlations
                        ],
                    }
                )

                above, below, assumptions_not_met = (
                    self.analyzer
                    ._categorize_bivariate_tests_as_rej_or_failrej(
                        test_df,
                        [(method, 0.6, None)],
                        check_assumptions=False,
                    )
                )

                self.assertEqual(
                    {row[1] for row in above},
                    {
                        "positive_strong",
                        "negative_strong",
                        "positive_boundary",
                        "negative_boundary",
                    },
                )
                self.assertEqual(
                    {row[1] for row in below},
                    {"positive_weak", "negative_weak"},
                )
                self.assertEqual(assumptions_not_met, [])

    def test_negative_relationship_updates_model_and_target_metadata(self):
        test_df = pd.DataFrame(
            {
                "column_a": ["x"],
                "column_b": ["y"],
                "test": ["spearman"],
                "Correlation": [-1.0],
            }
        )

        self.analyzer._update_model_with_test_df_to_col_pairs_and_cols_as_targets(
            test_df,
            targets=["x"],
            check_assumptions=False,
        )

        self.assertEqual(
            self.analyzer.above_threshold_corr_numnum,
            [["x", "y", "spearman"]],
        )
        self.assertEqual(self.analyzer.below_threshold_corr_numnum, [])
        self.assertEqual(
            self.analyzer.target_key_feature_meta_vals["x"][
                "significant_numeric_relationships"
            ],
            ["y"],
        )
        self.assertEqual(
            self.analyzer.target_key_feature_meta_vals["x"][
                "significant_numeric_tests"
            ],
            ["spearman"],
        )


class AnalyzeDatasetSingleTestFamilyTests(unittest.TestCase):
    def test_numeric_only_fit_without_targets_marks_numeric_columns_fitted(self):
        data = pd.DataFrame(
            {
                "x": list(range(20)),
                "y": [-2 * value for value in range(20)],
                "unselected": [value % 3 for value in range(20)],
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        result = analyzer.fit_column_relationships(
            data,
            numeric_columns=["x", "y"],
            check_assumptions=False,
        )

        self.assertIs(result, analyzer)
        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"x", "y"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"x", "y"},
        )
        self.assertEqual(
            analyzer.above_threshold_corr_numnum,
            [["x", "y", "pearson"]],
        )

    def test_categorical_only_fit_without_targets_marks_categories_fitted(self):
        data = pd.DataFrame(
            {
                "left": ["A"] * 20 + ["B"] * 20,
                "right": ["X"] * 20 + ["Y"] * 20,
                "unselected": ["U", "V", "W", "Z"] * 10,
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=[
                ("chi2", 0.05, None),
            ],
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        result = analyzer.fit_column_relationships(
            data,
            categoric_columns=["left", "right"],
            check_assumptions=False,
        )

        self.assertIs(result, analyzer)
        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"left", "right"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"left", "right"},
        )
        self.assertEqual(
            analyzer.reject_null_catcat,
            [["left", "right", "chi2"]],
        )

    def test_numcat_fit_uses_both_limited_column_pools_as_targets(self):
        data = pd.DataFrame(
            {
                "amount": list(range(40)),
                "unselected_numeric": [value % 5 for value in range(40)],
                "group": ["A"] * 20 + ["B"] * 20,
                "unselected_categoric": ["X", "Y"] * 20,
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=[
                ("kruskal", 0.05, None),
            ],
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["amount"],
            categoric_columns=["group"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"amount", "group"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"amount", "group"},
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["amount"]["target_dtype"],
            ["numeric"],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["group"]["target_dtype"],
            ["categoric"],
        )

    def test_explicit_target_is_added_outside_limited_feature_pool(self):
        data = pd.DataFrame(
            {
                "outcome": list(range(20)),
                "n1": [-2 * value for value in range(20)],
                "n2": [value % 3 for value in range(20)],
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["n1"],
            numeric_target="outcome",
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"outcome"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"outcome"},
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["outcome"][
                "significant_numeric_relationships"
            ],
            ["n1"],
        )
        self.assertNotIn(
            "n2",
            analyzer.target_key_feature_meta_vals["outcome"][
                "significant_numeric_relationships"
            ],
        )

    def test_explicit_categorical_target_is_added_outside_limited_feature_pool(
        self,
    ):
        data = pd.DataFrame(
            {
                "outcome": ["A"] * 20 + ["B"] * 20,
                "c1": ["X"] * 20 + ["Y"] * 20,
                "c2": ["U", "V"] * 20,
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=[
                ("chi2", 0.05, None),
            ],
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        analyzer.fit_column_relationships(
            data,
            categoric_columns=["c1"],
            categoric_target="outcome",
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"outcome"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"outcome"},
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["outcome"][
                "significant_categoric_relationships"
            ],
            ["c1"],
        )
        self.assertNotIn(
            "c2",
            analyzer.target_key_feature_meta_vals["outcome"][
                "significant_categoric_relationships"
            ],
        )

    def test_none_column_list_retains_numeric_autodetection(self):
        data = pd.DataFrame(
            {
                "x": list(range(20)),
                "y": [-2 * value for value in range(20)],
                "categoric": ["A", "B"] * 10,
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=None,
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"x", "y"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"x", "y"},
        )

    def test_selected_target_with_no_pair_gets_empty_metadata(self):
        data = pd.DataFrame(
            {
                "only": list(range(20)),
                "unselected": [value % 3 for value in range(20)],
            }
        )
        analyzer = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["only"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.has_called_fit_column_relationships,
            {"only"},
        )
        self.assertEqual(
            set(analyzer.target_key_feature_meta_vals),
            {"only"},
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["only"]["target_dtype"],
            ["numeric"],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["only"][
                "significant_numeric_relationships"
            ],
            [],
        )
        result = analyzer.column_relationships_df()
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
