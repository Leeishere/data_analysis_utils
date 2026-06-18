import unittest
from unittest.mock import patch

import pandas as pd

from data_analysis_utils import AnalyzeDataset
from data_analysis_utils.BinnerClass import Bin


class BinningMetricsTests(unittest.TestCase):
    @staticmethod
    def _numeric_analyzer():
        return AnalyzeDataset(
            numnum_meth_alpha_above_instructions=[
                ("pearson", 0.6, None),
            ],
            numcat_meth_alpha_above_instructions=False,
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=False,
        )

    @staticmethod
    def _numeric_data():
        return pd.DataFrame(
            {
                "target": list(range(30)),
                "feature": [value * 2 for value in range(30)],
            }
        )

    def test_relational_binner_accepts_correct_threshold_spelling(self):
        data = self._numeric_data()

        result = Bin().relational_binner(
            data,
            numnum_meth_alpha_above=("pearson", 0.6, True),
            numcat_meth_alpha_above=None,
            original_value_count_threshold=5,
            numeric_columns=["target", "feature"],
        )

        self.assertEqual(result, {"target": 2, "feature": 2})

    def test_relational_binner_preserves_deprecated_threshold_spelling(self):
        data = self._numeric_data()

        with self.assertWarns(DeprecationWarning):
            result = Bin().relational_binner(
                data,
                numnum_meth_alpha_above=("pearson", 0.6, True),
                numcat_meth_alpha_above=None,
                original_value_count_threashold=5,
                numeric_columns=["target", "feature"],
            )

        self.assertEqual(result, {"target": 2, "feature": 2})

    def test_numeric_relationship_returns_binning_metrics(self):
        data = self._numeric_data()
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["feature"],
            numeric_target="target",
            check_assumptions=False,
        )

        minimum, feature_map = analyzer.get_a_variables_binning_metrics(
            data,
            "target",
            original_value_count_threshold=5,
        )

        self.assertEqual(minimum, 2)
        self.assertEqual(feature_map, {"feature": 2})

    def test_deprecated_binning_metrics_method_remains_available(self):
        data = self._numeric_data()
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["feature"],
            numeric_target="target",
            check_assumptions=False,
        )

        with self.assertWarns(DeprecationWarning):
            minimum, feature_map = analyzer.get_a_varaibles_binning_metrics(
                data,
                "target",
                original_value_count_threshold=5,
            )

        self.assertEqual(minimum, 2)
        self.assertEqual(feature_map, {"feature": 2})

    def test_threshold_controls_target_eligibility(self):
        data = self._numeric_data()
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["feature"],
            numeric_target="target",
            check_assumptions=False,
        )

        minimum, feature_map = analyzer.get_a_variables_binning_metrics(
            data,
            "target",
            original_value_count_threshold=30,
        )

        self.assertIsNone(minimum)
        self.assertEqual(feature_map, {})

    def test_numeric_categorical_path_ignores_disabled_numeric_tests(self):
        data = pd.DataFrame(
            {
                "target": list(range(40)),
                "group": ["A"] * 20 + ["B"] * 20,
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
            categoric_columns=["group"],
            numeric_target="target",
            check_assumptions=False,
        )

        minimum, feature_map = analyzer.get_a_variables_binning_metrics(
            data,
            "target",
        )

        self.assertIsNone(minimum)
        self.assertEqual(feature_map, {"group": None})

    def test_fit_thresholds_selects_numeric_targets_and_is_repeatable(self):
        data = self._numeric_data()
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["feature"],
            numeric_target="target",
            check_assumptions=False,
        )
        category_meta = analyzer._blank_target_dict()
        category_meta["target_dtype"] = ["categoric"]
        analyzer.target_key_feature_meta_vals["category"] = category_meta

        first_result = analyzer.fit_binning_thresholds(data)
        second_result = analyzer.fit_binning_thresholds(data)

        self.assertIs(first_result, analyzer)
        self.assertIs(second_result, analyzer)
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["target"]["min_bins"],
            [2],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["target"][
                "min_bins_by_feature"
            ],
            {"feature": 2},
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["category"]["min_bins"],
            [],
        )

    def test_none_threshold_is_stored_and_repeated_safely(self):
        data = pd.DataFrame({"only": list(range(30))})
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["only"],
            check_assumptions=False,
        )

        analyzer.fit_binning_thresholds(data)
        analyzer.fit_binning_thresholds(data)

        self.assertEqual(
            analyzer.target_key_feature_meta_vals["only"]["min_bins"],
            [None],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["only"][
                "min_bins_by_feature"
            ],
            {},
        )

    def test_explicit_target_selection_updates_only_requested_target(self):
        data = pd.DataFrame(
            {
                "first": list(range(30)),
                "second": [value * 2 for value in range(30)],
            }
        )
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["first", "second"],
            check_assumptions=False,
        )

        analyzer.fit_binning_thresholds(data, targets="first")

        self.assertEqual(
            analyzer.target_key_feature_meta_vals["first"]["min_bins"],
            [2],
        )
        self.assertEqual(
            analyzer.target_key_feature_meta_vals["second"]["min_bins"],
            [],
        )

    def test_forced_plot_bins_use_repaired_metrics_path(self):
        data = self._numeric_data()
        analyzer = self._numeric_analyzer()
        analyzer.fit_column_relationships(
            data,
            numeric_columns=["feature"],
            numeric_target="target",
            check_assumptions=False,
        )

        with patch.object(
            analyzer,
            "univariate_numerical_snapshot",
        ) as snapshot:
            result = analyzer.plot_non_normal_numeric(
                data,
                numerical=["target"],
                force_significant_bin_edges=True,
                minimize_significant_bins=True,
            )

        self.assertIs(result, analyzer)
        keep_bins = snapshot.call_args.kwargs["keep_bins_significant"]
        self.assertEqual(set(keep_bins), {"target"})
        self.assertEqual(len(keep_bins["target"]), 3)


if __name__ == "__main__":
    unittest.main()
