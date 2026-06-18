import unittest
from unittest.mock import patch

import pandas as pd

from data_analysis_utils import AnalyzeDataset


class FullDatasetAnalysisTests(unittest.TestCase):
    @staticmethod
    def _analyzer():
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
    def _data():
        return pd.DataFrame(
            {
                "x": list(range(20)),
                "y": [-2 * value for value in range(20)],
            }
        )

    def test_returns_same_analyzer_with_optional_stages_disabled(self):
        analyzer = self._analyzer()

        result = analyzer.fit_full_dataset_analysis(
            self._data(),
            numeric_columns=["x", "y"],
            fit_good_of_fit=False,
            fit_normal=False,
            fit_multivariates=False,
            fit_supercat_subcats=False,
            check_assumptions=False,
        )

        self.assertIs(result, analyzer)
        self.assertEqual(
            analyzer.above_threshold_corr_numnum,
            [["x", "y", "pearson"]],
        )

    def test_return_value_supports_relationship_dataframe_chaining(self):
        analyzer = self._analyzer()

        relationships = analyzer.fit_full_dataset_analysis(
            self._data(),
            numeric_columns=["x", "y"],
            fit_good_of_fit=False,
            fit_normal=False,
            fit_multivariates=False,
            fit_supercat_subcats=False,
            check_assumptions=False,
        ).column_relationships_df()

        self.assertFalse(relationships.empty)
        self.assertEqual(
            set(relationships.index.get_level_values("Target")),
            {"x", "y"},
        )

    def test_returns_self_after_enabled_optional_stages(self):
        analyzer = self._analyzer()

        with patch.object(
            analyzer,
            "fit_multivariate_column_relationships",
            return_value=analyzer,
        ) as fit_multivariate, patch.object(
            analyzer,
            "fit_supercat_subcat_pairs",
            return_value=analyzer,
        ) as fit_supercat:
            result = analyzer.fit_full_dataset_analysis(
                self._data(),
                numeric_columns=["x", "y"],
                fit_good_of_fit=False,
                fit_normal=False,
                fit_multivariates=True,
                fit_supercat_subcats=True,
                check_assumptions=False,
            )

        self.assertIs(result, analyzer)
        fit_multivariate.assert_called_once()
        fit_supercat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
