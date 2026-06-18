import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import chi2_contingency

from data_analysis_utils import AnalyzeDataset
from data_analysis_utils.Chi2 import Chi2


class Chi2Tests(unittest.TestCase):
    def setUp(self):
        self.chi2 = Chi2()
        self.sparse = pd.DataFrame(
            {
                "left": ["A", "A", "B", "B"],
                "right": ["X", "X", "Y", "Y"],
            }
        )

    def test_frequency_table_uses_zero_filled_unstack(self):
        result = self.chi2.frequencies_table(
            self.sparse,
            "left",
            "right",
            kind="frequency",
        )
        expected = pd.DataFrame(
            [[2, 0], [0, 2]],
            index=pd.Index(["A", "B"], name="left"),
            columns=pd.Index(["X", "Y"], name="right"),
        )

        assert_frame_equal(result, expected)

    def test_joint_probability_table_is_zero_filled_and_sums_to_one(self):
        result = self.chi2.frequencies_table(
            self.sparse,
            "left",
            "right",
            kind="joint_probability",
        )

        self.assertEqual(result.loc["A", "Y"], 0.0)
        self.assertEqual(result.loc["B", "X"], 0.0)
        self.assertAlmostEqual(result.to_numpy().sum(), 1.0)

    def test_contingency_table_preserves_margins_and_dtypes(self):
        result = self.chi2.contingency_table(
            self.sparse,
            "left",
            "right",
            kind="frequency",
        )
        expected = pd.DataFrame(
            [[2, 0, 2], [0, 2, 2], [2, 2, 4]],
            index=pd.Index(["A", "B", "BOTTOM_MARGIN"], name="left"),
            columns=pd.Index(
                ["X", "Y", "RIGHT_MARGIN"],
                name="right",
            ),
        )

        assert_frame_equal(result, expected)

    def test_subcategories_map_to_most_common_supercategories(self):
        data = pd.DataFrame(
            {
                "super": ["B", "A", "A", "B", "B", "A", "B"],
                "sub": ["z", "x", "x", "z", "y", "y", "y"],
            }
        )

        result = self.chi2.map_subcat_to_supercat(
            data,
            supercat="super",
            subcat="sub",
        )
        expected = pd.DataFrame(
            {
                "sub": ["x", "y", "z"],
                "super": ["A", "B", "B"],
            }
        )

        assert_frame_equal(result, expected)

    def test_subcategory_mapping_is_independent_of_row_order(self):
        data = pd.DataFrame(
            {
                "super": ["B", "A", "A", "B", "B", "A", "B"],
                "sub": ["z", "x", "x", "z", "y", "y", "y"],
            }
        )

        forward = self.chi2.map_subcat_to_supercat(
            data,
            supercat="super",
            subcat="sub",
        )
        reversed_result = self.chi2.map_subcat_to_supercat(
            data.iloc[::-1],
            supercat="super",
            subcat="sub",
        )

        assert_frame_equal(forward, reversed_result)

    def test_subcategory_mapping_handles_missing_values(self):
        data = pd.DataFrame(
            {
                "super": ["A", np.nan, np.nan, "B"],
                "sub": ["x", "missing-super", "missing-super", np.nan],
            }
        )

        result = self.chi2.map_subcat_to_supercat(
            data,
            supercat="super",
            subcat="sub",
        )

        self.assertEqual(list(result.columns), ["sub", "super"])
        self.assertEqual(len(result), 3)
        missing_super = result.loc[
            result["sub"] == "missing-super",
            "super",
        ].iloc[0]
        self.assertTrue(pd.isna(missing_super))
        missing_sub = result.loc[result["sub"].isna(), "super"].iloc[0]
        self.assertEqual(missing_sub, "B")

    def test_subcategory_mapping_ignores_unused_categorical_levels(self):
        data = pd.DataFrame(
            {
                "super": pd.Categorical(
                    ["A", "A", "B"],
                    categories=["A", "B", "unused-super"],
                ),
                "sub": pd.Categorical(
                    ["x", "x", "y"],
                    categories=["x", "y", "unused-sub"],
                ),
            }
        )

        result = self.chi2.map_subcat_to_supercat(
            data,
            supercat="super",
            subcat="sub",
        )

        self.assertEqual(result["sub"].tolist(), ["x", "y"])
        self.assertEqual(result["super"].tolist(), ["A", "B"])

    def test_subcategory_mapping_ties_follow_value_counts_order(self):
        data = pd.DataFrame(
            {
                "super": ["B", "A"],
                "sub": ["tied", "tied"],
            }
        )
        expected_super = (
            data["super"]
            .value_counts(dropna=False)
            .idxmax()
        )

        result = self.chi2.map_subcat_to_supercat(
            data,
            supercat="super",
            subcat="sub",
        )

        self.assertEqual(result.loc[0, "super"], expected_super)

    def test_sparse_independence_matches_scipy_reference(self):
        observed = np.array([[2, 0], [0, 2]])
        expected = chi2_contingency(
            observed,
            correction=False,
        ).pvalue
        result = self.chi2.chi_squared_independence(
            self.sparse,
            "left",
            "right",
            check_assumptions=False,
        )

        self.assertAlmostEqual(result, expected)
        self.assertAlmostEqual(result, 0.04550026389635857)

    def test_dense_tables_match_scipy_reference(self):
        data = pd.DataFrame(
            {
                "left": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
                "right": [
                    "X", "X", "X", "Y", "Y", "Z",
                    "X", "Y", "Y", "Y", "Z", "Z",
                    "X", "X", "Y", "Z", "Z", "Z",
                ],
            }
        )
        observed = (
            data.groupby(["left", "right"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        expected = chi2_contingency(
            observed,
            correction=False,
        ).pvalue
        result = self.chi2.chi_squared_independence(
            data,
            "left",
            "right",
            check_assumptions=False,
        )

        self.assertAlmostEqual(result, expected)

    def test_row_and_category_order_do_not_change_p_value(self):
        reversed_data = self.sparse.iloc[::-1].copy()
        reversed_data["left"] = pd.Categorical(
            reversed_data["left"],
            categories=["B", "A"],
            ordered=True,
        )
        reversed_data["right"] = pd.Categorical(
            reversed_data["right"],
            categories=["Y", "X"],
            ordered=True,
        )

        original = self.chi2.chi_squared_independence(
            self.sparse,
            "left",
            "right",
            check_assumptions=False,
        )
        reordered = self.chi2.chi_squared_independence(
            reversed_data,
            "left",
            "right",
            check_assumptions=False,
        )

        self.assertAlmostEqual(reordered, original)

    def test_dropna_false_treats_missing_values_as_categories(self):
        data = pd.DataFrame(
            {
                "left": ["A", None, "B", None, "A", "B"],
                "right": ["X", "Y", None, "Y", None, "X"],
            }
        )
        transformed = data.where(~data.isna(), "NaN")
        observed = (
            transformed.groupby(["left", "right"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        expected = chi2_contingency(
            observed,
            correction=False,
        ).pvalue
        result = self.chi2.chi_squared_independence(
            data,
            "left",
            "right",
            check_assumptions=False,
            dropna=False,
        )

        self.assertAlmostEqual(result, expected)

    def test_unused_categorical_levels_do_not_change_test(self):
        categorical_data = pd.DataFrame(
            {
                "left": pd.Categorical(
                    self.sparse["left"],
                    categories=["A", "B", "UNUSED_LEFT"],
                ),
                "right": pd.Categorical(
                    self.sparse["right"],
                    categories=["X", "Y", "UNUSED_RIGHT"],
                ),
            }
        )
        public_table = self.chi2.frequencies_table(
            categorical_data,
            "left",
            "right",
            kind="frequency",
        )
        result = self.chi2.chi_squared_independence(
            categorical_data,
            "left",
            "right",
            check_assumptions=False,
        )

        self.assertIn("UNUSED_LEFT", public_table.index)
        self.assertIn("UNUSED_RIGHT", public_table.columns)
        self.assertEqual(public_table.loc["UNUSED_LEFT"].sum(), 0)
        self.assertEqual(public_table["UNUSED_RIGHT"].sum(), 0)
        self.assertAlmostEqual(result, 0.04550026389635857)

    def test_sparse_expected_counts_fail_assumption_check(self):
        p_value, assumptions_met = self.chi2.chi_squared_independence(
            self.sparse,
            "left",
            "right",
            check_assumptions=True,
        )

        self.assertAlmostEqual(p_value, 0.04550026389635857)
        self.assertFalse(assumptions_met)

    def test_degenerate_inputs_return_nan_and_false(self):
        cases = [
            pd.DataFrame({"left": [], "right": []}),
            pd.DataFrame(
                {
                    "left": ["A", "A", "A"],
                    "right": ["X", "Y", "Y"],
                }
            ),
            pd.DataFrame(
                {
                    "left": ["A", "B", "B"],
                    "right": ["X", "X", "X"],
                }
            ),
        ]

        for data in cases:
            with self.subTest(data=data):
                p_value = self.chi2.chi_squared_independence(
                    data,
                    "left",
                    "right",
                    check_assumptions=False,
                )
                checked_p_value, assumptions_met = (
                    self.chi2.chi_squared_independence(
                        data,
                        "left",
                        "right",
                        check_assumptions=True,
                    )
                )

                self.assertTrue(np.isnan(p_value))
                self.assertTrue(np.isnan(checked_p_value))
                self.assertFalse(assumptions_met)

    def test_batch_and_comparison_schemas_remain_stable(self):
        batch = self.chi2.test_all_cat_columns_chi_independence(
            self.sparse,
            columns=["left", "right"],
            check_assumptions=True,
        )
        rejected = self.chi2.categorical_column_comparison(
            self.sparse,
            alpha=0.05,
            keep_above_p=False,
            categoric_columns=["left", "right"],
            check_assumptions=False,
        )

        self.assertEqual(
            batch.columns.tolist(),
            [
                "category_a",
                "category_b",
                "P-value",
                "assumptions_met",
            ],
        )
        self.assertFalse(batch.loc[0, "assumptions_met"])
        self.assertEqual(
            rejected[["category_a", "category_b"]].values.tolist(),
            [["left", "right"]],
        )


class AnalyzeDatasetChi2IntegrationTests(unittest.TestCase):
    @staticmethod
    def _analyzer(check_assumptions):
        return AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=True,
            catcat_meth_alpha_above_instructions=[
                ("chi2", 0.05, None),
            ],
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=check_assumptions,
        )

    def test_sparse_pair_rejects_when_assumptions_are_disabled(self):
        data = pd.DataFrame(
            {
                "left": ["A", "A", "B", "B"],
                "right": ["X", "X", "Y", "Y"],
            }
        )
        analyzer = self._analyzer(check_assumptions=False)

        analyzer.fit_column_relationships(
            data,
            categoric_columns=["left", "right"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.reject_null_catcat,
            [["left", "right", "chi2"]],
        )
        self.assertEqual(analyzer.fail_to_reject_null_catcat, [])
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["left"][
                "significant_categoric_relationships"
            ],
            ["right"],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["right"][
                "significant_categoric_relationships"
            ],
            ["left"],
        )

        pairs, is_super_subcategory = analyzer._are_supercat_subcats(data)
        self.assertEqual(pairs, [["left", "right"]])
        self.assertEqual(is_super_subcategory, [True])

    def test_sparse_pair_is_only_assumptions_not_met_when_checked(self):
        data = pd.DataFrame(
            {
                "left": ["A", "A", "B", "B"],
                "right": ["X", "X", "Y", "Y"],
            }
        )
        analyzer = self._analyzer(check_assumptions=True)

        analyzer.fit_column_relationships(
            data,
            categoric_columns=["left", "right"],
            check_assumptions=True,
        )

        self.assertEqual(analyzer.reject_null_catcat, [])
        self.assertEqual(analyzer.fail_to_reject_null_catcat, [])
        self.assertEqual(
            analyzer.assumptions_not_met["catcat"],
            [["left", "right"]],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["left"][
                "assumptions_not_met"
            ]["catcat"],
            ["right"],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["right"][
                "assumptions_not_met"
            ]["catcat"],
            ["left"],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["left"][
                "significant_categoric_relationships"
            ],
            [],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["left"][
                "not_significant_categorics"
            ],
            [],
        )

    def test_undefined_pair_is_unclassified_without_assumption_checks(self):
        analyzer = self._analyzer(check_assumptions=False)
        test_df = pd.DataFrame(
            {
                "column_a": ["left"],
                "column_b": ["right"],
                "test": ["chi2"],
                "P-value": [np.nan],
            }
        )

        rejected, failed, assumptions_not_met = (
            analyzer._categorize_bivariate_tests_as_rej_or_failrej(
                test_df,
                [("chi2", 0.05, None)],
                check_assumptions=False,
            )
        )

        self.assertEqual(rejected, [])
        self.assertEqual(failed, [])
        self.assertEqual(assumptions_not_met, [])

    def test_undefined_pair_is_only_assumptions_not_met_when_checked(self):
        analyzer = self._analyzer(check_assumptions=True)
        test_df = pd.DataFrame(
            {
                "column_a": ["left"],
                "column_b": ["right"],
                "test": ["chi2"],
                "P-value": [np.nan],
                "assumptions_met": [False],
            }
        )

        rejected, failed, assumptions_not_met = (
            analyzer._categorize_bivariate_tests_as_rej_or_failrej(
                test_df,
                [("chi2", 0.05, None)],
                check_assumptions=True,
            )
        )

        self.assertEqual(rejected, [])
        self.assertEqual(failed, [])
        self.assertEqual(assumptions_not_met, [["left", "right"]])


if __name__ == "__main__":
    unittest.main()
