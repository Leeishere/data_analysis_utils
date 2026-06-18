import unittest

import pandas as pd

from data_analysis_utils import AnalyzeDataset


class AnalyzeDatasetInstructionNormalizationTests(unittest.TestCase):
    @staticmethod
    def _analyzer(**overrides):
        settings = {
            "numnum_meth_alpha_above_instructions": False,
            "numcat_meth_alpha_above_instructions": False,
            "catcat_meth_alpha_above_instructions": False,
            "good_of_fit_uniform_test_instructions": False,
            "normal_test_instructions": False,
            "check_assumptions": False,
        }
        settings.update(overrides)
        return AnalyzeDataset(**settings)

    def test_direct_numeric_tuple_is_normalized_and_fitted(self):
        data = pd.DataFrame(
            {
                "x": list(range(20)),
                "y": [-2 * value for value in range(20)],
            }
        )
        analyzer = self._analyzer(
            numnum_meth_alpha_above_instructions=(
                "pearson",
                0.6,
                None,
            )
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["x", "y"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.numnum_meth_alpha_above,
            [("pearson", 0.6, None)],
        )
        self.assertEqual(
            analyzer.above_threshold_corr_numnum,
            [["x", "y", "pearson"]],
        )

    def test_direct_numcat_tuple_is_normalized_and_fitted(self):
        data = pd.DataFrame(
            {
                "value": list(range(40)),
                "group": ["A"] * 20 + ["B"] * 20,
            }
        )
        analyzer = self._analyzer(
            numcat_meth_alpha_above_instructions=(
                "kruskal",
                0.05,
                None,
            )
        )

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["value"],
            categoric_columns=["group"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.numcat_meth_alpha_above,
            [("kruskal", 0.05, None)],
        )
        self.assertEqual(
            analyzer.reject_null_numcat,
            [["value", "group", "kruskal"]],
        )

    def test_direct_categorical_tuple_is_normalized_and_fitted(self):
        data = pd.DataFrame(
            {
                "left": ["A"] * 20 + ["B"] * 20,
                "right": ["X"] * 20 + ["Y"] * 20,
            }
        )
        analyzer = self._analyzer(
            catcat_meth_alpha_above_instructions=(
                "chi2",
                0.05,
                None,
            )
        )

        analyzer.fit_column_relationships(
            data,
            categoric_columns=["left", "right"],
            check_assumptions=False,
        )

        self.assertEqual(
            analyzer.catcat_meth_alpha_above,
            [("chi2", 0.05, None)],
        )
        self.assertEqual(
            analyzer.reject_null_catcat,
            [["left", "right", "chi2"]],
        )

    def test_direct_list_and_nested_sequences_share_canonical_shape(self):
        cases = (
            ["chi2", 0.05, None],
            [["chi2", 0.05, None]],
            [("chi2", 0.05, None)],
            (("chi2", 0.05, None),),
        )

        for instructions in cases:
            with self.subTest(instructions=instructions):
                analyzer = self._analyzer(
                    catcat_meth_alpha_above_instructions=instructions
                )
                self.assertEqual(
                    analyzer.catcat_meth_alpha_above,
                    [("chi2", 0.05, None)],
                )

    def test_boolean_and_none_configuration_semantics_are_preserved(self):
        defaults = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=True,
            numcat_meth_alpha_above_instructions=True,
            catcat_meth_alpha_above_instructions=True,
        )
        disabled = AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=None,
            catcat_meth_alpha_above_instructions=False,
        )

        self.assertEqual(
            defaults.numnum_meth_alpha_above,
            [
                ("pearson", 0.6, None),
                ("spearman", 0.6, None),
                ("kendall", 0.6, None),
            ],
        )
        self.assertEqual(
            defaults.numcat_meth_alpha_above,
            [("kruskal", 0.05, None), ("anova", 0.05, None)],
        )
        self.assertEqual(
            defaults.catcat_meth_alpha_above,
            [("chi2", 0.05, None)],
        )
        self.assertIsNone(disabled.numnum_meth_alpha_above)
        self.assertIsNone(disabled.numcat_meth_alpha_above)
        self.assertIsNone(disabled.catcat_meth_alpha_above)


if __name__ == "__main__":
    unittest.main()
