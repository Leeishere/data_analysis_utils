import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import scipy.stats

from data_analysis_utils import AnalyzeDataset
from data_analysis_utils.ANOVA import ANOVA


def scipy_kruskal_p_value(data):
    groups = [
        group.to_numpy()
        for _, group in data.groupby("group", observed=True)["value"]
    ]
    return scipy.stats.kruskal(*groups).pvalue


class KruskalAssumptionTests(unittest.TestCase):
    def setUp(self):
        self.anova = ANOVA()
        self.same_distributions = pd.DataFrame(
            {
                "group": ["a"] * 100 + ["b"] * 100,
                "value": list(np.linspace(-1, 1, 100)) * 2,
            }
        )
        self.different_distributions = pd.DataFrame(
            {
                "group": ["a"] * 100 + ["b"] * 100,
                "value": (
                    list(np.linspace(-1, 1, 100))
                    + list(np.linspace(5, 7, 100))
                ),
            }
        )

    def test_pairwise_ks_true_means_similarity_criteria_passed(self):
        same_result = self.anova._ks_pairwise_parallel(
            self.same_distributions,
            "group",
            "value",
            n_jobs=1,
            alpha=0.05,
            return_meta=False,
            guesstimate=False,
        )
        different_result = self.anova._ks_pairwise_parallel(
            self.different_distributions,
            "group",
            "value",
            n_jobs=1,
            alpha=0.05,
            return_meta=False,
            guesstimate=False,
        )

        self.assertTrue(same_result)
        self.assertFalse(different_result)

    def test_guesstimate_can_tolerate_configured_pairwise_rejections(self):
        strict_result = self.anova._ks_pairwise_parallel(
            self.different_distributions,
            "group",
            "value",
            n_jobs=1,
            alpha=0.05,
            return_meta=False,
            guesstimate=False,
        )
        tolerant_result = self.anova._ks_pairwise_parallel(
            self.different_distributions,
            "group",
            "value",
            n_jobs=1,
            alpha=0.05,
            return_meta=False,
            guesstimate={
                "rej_max_pct_in_group": 1.0,
                "max_num_outlier_all_reject": 3,
                "max_pct_reject_total": 1.0,
            },
        )

        self.assertFalse(strict_result)
        self.assertTrue(tolerant_result)

    def test_parallel_failure_retries_with_all_available_workers(self):
        failed_runner = Mock(side_effect=RuntimeError("invalid worker count"))
        retry_runner = Mock(return_value=[("a", "b", 0.0, 1.0)])

        with patch(
            "data_analysis_utils.ANOVA.Parallel",
            side_effect=[failed_runner, retry_runner],
        ) as parallel:
            with self.assertWarns(UserWarning):
                result = self.anova._ks_pairwise_parallel(
                    self.same_distributions,
                    "group",
                    "value",
                    n_jobs=999,
                    alpha=0.05,
                    return_meta=False,
                    guesstimate=False,
                )

        self.assertTrue(result)
        self.assertEqual(
            parallel.call_args_list,
            [call(n_jobs=999), call(n_jobs=-1)],
        )

    @patch("data_analysis_utils.ANOVA.scipy.stats.levene")
    def test_strict_path_accepts_similar_and_rejects_different_distributions(
        self,
        mocked_levene,
    ):
        mocked_levene.return_value = (0.0, 1.0)

        same_result = self.anova.kruskal_wallis_assumptions(
            self.same_distributions["group"],
            self.same_distributions["value"],
            retrieve_meta=False,
            return_pseudo=False,
            guesstimate=False,
            n_jobs=1,
        )
        different_result = self.anova.kruskal_wallis_assumptions(
            self.different_distributions["group"],
            self.different_distributions["value"],
            retrieve_meta=False,
            return_pseudo=False,
            guesstimate=False,
            n_jobs=1,
        )

        self.assertTrue(same_result)
        self.assertFalse(different_result)

    @patch("data_analysis_utils.ANOVA.scipy.stats.levene")
    def test_strict_ks_failure_can_return_pseudo_result(
        self,
        mocked_levene,
    ):
        mocked_levene.return_value = (0.0, 1.0)

        result = self.anova.kruskal_wallis_assumptions(
            self.different_distributions["group"],
            self.different_distributions["value"],
            retrieve_meta=False,
            return_pseudo=True,
            pseudo_test_max_global_ties_ratio=0.5,
            guesstimate=False,
            n_jobs=1,
        )

        self.assertEqual(result, "Psuedo")

    def test_full_pseudo_still_bypasses_strict_checks(self):
        result = self.anova.kruskal_wallis_assumptions(
            self.different_distributions["group"],
            self.different_distributions["value"],
            retrieve_meta=False,
            full_pseudo=True,
            pseudo_test_max_global_ties_ratio=0.5,
        )

        self.assertEqual(result, "Psuedo")

    def test_wrapper_forwards_strict_parameters(self):
        with patch.object(
            self.anova,
            "_kruskal_assumptions_strict",
            return_value=True,
        ) as strict:
            result = self.anova.kruskal_wallis_assumptions(
                self.same_distributions["group"],
                self.same_distributions["value"],
                retrieve_meta=False,
                guesstimate={"max_pct_reject_total": 0.4},
                n_jobs=-2,
            )

        self.assertTrue(result)
        self.assertEqual(
            strict.call_args.kwargs["guesstimate"],
            {"max_pct_reject_total": 0.4},
        )
        self.assertEqual(strict.call_args.kwargs["n_jobs"], -2)

    def test_wrapper_forwards_metadata_parameters(self):
        expected = {"ks_pairwise": pd.DataFrame()}
        with patch.object(
            self.anova,
            "_kruskal_assumptions_meta",
            return_value=expected,
        ) as metadata:
            result = self.anova.kruskal_wallis_assumptions(
                self.same_distributions["group"],
                self.same_distributions["value"],
                retrieve_meta=True,
                guesstimate={"max_pct_reject_total": 0.4},
                n_jobs=-2,
            )

        self.assertIs(result, expected)
        self.assertEqual(
            metadata.call_args.kwargs["guesstimate"],
            {"max_pct_reject_total": 0.4},
        )
        self.assertEqual(metadata.call_args.kwargs["n_jobs"], -2)


class KruskalPValueTests(unittest.TestCase):
    def setUp(self):
        self.anova = ANOVA()

    def test_tied_data_matches_scipy_tie_corrected_result(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
            }
        )
        expected = scipy_kruskal_p_value(data)
        result = self.anova.one_way_kruskal_wallis(data)

        self.assertAlmostEqual(result, expected)

    def test_tie_free_data_remains_equivalent_to_scipy(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": list(range(1, 11)),
            }
        )
        expected = scipy_kruskal_p_value(data)
        result = self.anova.one_way_kruskal_wallis(data)

        self.assertAlmostEqual(result, expected)

    def test_all_identical_values_return_nan(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1] * 10,
            }
        )

        result = self.anova.one_way_kruskal_wallis(data)

        self.assertTrue(np.isnan(result))

    def test_nonfinite_p_value_forces_assumptions_false(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1] * 10,
            }
        )

        with patch.object(
            self.anova,
            "kruskal_wallis_assumptions",
            return_value="Psuedo",
        ) as assumptions:
            result = self.anova.test_all_num_cat_kruskal_wallis(
                data,
                numeric_columns=["value"],
                categoric_columns=["group"],
                check_assumptions=True,
                assumption_check_params={
                    "return_pseudo": True,
                    "n_jobs": 1,
                },
            )

        self.assertTrue(np.isnan(result.loc[0, "P-value"]))
        self.assertFalse(result.loc[0, "assumptions_met"])
        assumptions.assert_not_called()


class AnalyzeDatasetKruskalIntegrationTests(unittest.TestCase):
    @staticmethod
    def _analyzer(return_pseudo=False):
        return AnalyzeDataset(
            numnum_meth_alpha_above_instructions=False,
            numcat_meth_alpha_above_instructions=[
                ("kruskal", 0.05, None),
            ],
            catcat_meth_alpha_above_instructions=False,
            good_of_fit_uniform_test_instructions=False,
            normal_test_instructions=False,
            check_assumptions=True,
            kruskal_assumption_check_params={
                "levene_alpha": 0.05,
                "ks_alpha": 0.05,
                "return_pseudo": return_pseudo,
                "full_pseudo": False,
                "dropna": True,
                "guesstimate": False,
                "n_jobs": 1,
            },
        )

    def test_valid_kruskal_result_is_classified_normally(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 100 + ["b"] * 100,
                "value": list(np.linspace(-1, 1, 100)) * 2,
            }
        )
        analyzer = self._analyzer()

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["value"],
            categoric_columns=["group"],
            check_assumptions=True,
        )

        self.assertEqual(analyzer.reject_null_numcat, [])
        self.assertEqual(
            analyzer.fail_to_reject_null_numcat,
            [["value", "group"]],
        )
        self.assertEqual(analyzer.assumptions_not_met["numcat"], [])

    @patch("data_analysis_utils.ANOVA.scipy.stats.levene")
    def test_invalid_kruskal_result_is_only_assumptions_not_met(
        self,
        mocked_levene,
    ):
        mocked_levene.return_value = (0.0, 1.0)
        data = pd.DataFrame(
            {
                "group": ["a"] * 100 + ["b"] * 100,
                "value": (
                    list(np.linspace(-1, 1, 100))
                    + list(np.linspace(5, 7, 100))
                ),
            }
        )
        analyzer = self._analyzer()

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["value"],
            categoric_columns=["group"],
            check_assumptions=True,
        )

        self.assertEqual(analyzer.reject_null_numcat, [])
        self.assertEqual(analyzer.fail_to_reject_null_numcat, [])
        self.assertEqual(
            analyzer.assumptions_not_met["numcat"],
            [["value", "group"]],
        )

    @patch("data_analysis_utils.ANOVA.scipy.stats.levene")
    def test_pseudo_result_remains_a_valid_classification(
        self,
        mocked_levene,
    ):
        mocked_levene.return_value = (0.0, 1.0)
        data = pd.DataFrame(
            {
                "group": ["a"] * 100 + ["b"] * 100,
                "value": (
                    list(np.linspace(-1, 1, 100))
                    + list(np.linspace(5, 7, 100))
                ),
            }
        )
        analyzer = self._analyzer(return_pseudo=True)

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["value"],
            categoric_columns=["group"],
            check_assumptions=True,
        )

        self.assertEqual(
            analyzer.reject_null_numcat,
            [["value", "group", "kruskal:Psuedo"]],
        )
        self.assertEqual(analyzer.fail_to_reject_null_numcat, [])
        self.assertEqual(analyzer.assumptions_not_met["numcat"], [])

    def test_identical_values_are_only_assumptions_not_met(self):
        data = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1] * 10,
            }
        )
        analyzer = self._analyzer(return_pseudo=True)

        analyzer.fit_column_relationships(
            data,
            numeric_columns=["value"],
            categoric_columns=["group"],
            check_assumptions=True,
        )

        self.assertEqual(analyzer.reject_null_numcat, [])
        self.assertEqual(analyzer.fail_to_reject_null_numcat, [])
        self.assertEqual(
            analyzer.assumptions_not_met["numcat"],
            [["value", "group"]],
        )

    def test_nonfinite_p_value_overrides_pseudo_assumption_status(self):
        analyzer = self._analyzer(return_pseudo=True)
        test_df = pd.DataFrame(
            {
                "column_a": ["value"],
                "column_b": ["group"],
                "test": ["kruskal"],
                "P-value": [np.nan],
                "assumptions_met": ["Psuedo"],
            }
        )

        rejected, failed, assumptions_not_met = (
            analyzer._categorize_bivariate_tests_as_rej_or_failrej(
                test_df,
                [("kruskal", 0.05, None)],
                check_assumptions=True,
            )
        )

        self.assertEqual(rejected, [])
        self.assertEqual(failed, [])
        self.assertEqual(assumptions_not_met, [["value", "group"]])


if __name__ == "__main__":
    unittest.main()
