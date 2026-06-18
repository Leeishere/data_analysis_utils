import copy
import unittest
from unittest.mock import patch

import pandas as pd

from data_analysis_utils import AnalyzeDataset


class SupercategoryMetadataTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "super": ["S1", "S1", "S1", "S2", "S2", "S2"],
                "sub": ["a", "b", "c", "d", "e", "f"],
            }
        )

    @staticmethod
    def _analyzer(targets=("super", "sub")):
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
        analyzer.reject_null_catcat = [["sub", "super", "chi2"]]
        analyzer.has_called_fit_column_relationships = set(targets)

        if "super" in targets:
            super_meta = analyzer._blank_target_dict()
            super_meta["target_dtype"] = ["categoric"]
            super_meta["significant_categoric_relationships"] = ["sub"]
            super_meta["significant_categoric_tests"] = ["chi2"]
            analyzer.target_key_feature_meta_vals["super"] = super_meta

        if "sub" in targets:
            sub_meta = analyzer._blank_target_dict()
            sub_meta["target_dtype"] = ["categoric"]
            sub_meta["significant_categoric_relationships"] = ["super"]
            sub_meta["significant_categoric_tests"] = ["chi2"]
            analyzer.target_key_feature_meta_vals["sub"] = sub_meta

        return analyzer

    def test_failed_pair_preserves_original_orientation(self):
        analyzer = self._analyzer()
        data = pd.DataFrame(
            {
                "left": ["A", "A", "B", "B", "C", "C"],
                "right": ["X", "Y", "X", "Y", "X", "Y"],
            }
        )

        pairs, outcomes = analyzer._are_supercat_subcats(
            data,
            max_evidence=0.2,
            pairs_list=[["left", "right"]],
        )

        self.assertEqual(pairs, [["left", "right"]])
        self.assertEqual(outcomes, [False])

    def test_failed_pair_only_records_that_it_was_tested(self):
        for isolate in (False, True):
            with self.subTest(isolate_super_subs=isolate):
                analyzer = self._analyzer()
                original_reject = copy.deepcopy(analyzer.reject_null_catcat)
                original_metadata = copy.deepcopy(
                    analyzer.target_key_feature_meta_vals
                )

                with patch.object(
                    analyzer,
                    "_are_supercat_subcats",
                    return_value=([["super", "sub"]], [False]),
                ):
                    analyzer.fit_supercat_subcat_pairs(
                        self.data,
                        pairs_list=[["super", "sub"]],
                        isolate_super_subs=isolate,
                    )

                self.assertEqual(analyzer.supercategory_subcategory_pairs, [])
                self.assertEqual(
                    analyzer.has_called_fit_supercat_subcat_pairs,
                    [["sub", "super"]],
                )
                self.assertEqual(analyzer.reject_null_catcat, original_reject)
                self.assertEqual(
                    analyzer.target_key_feature_meta_vals,
                    original_metadata,
                )

    def test_successful_pair_updates_metadata_without_isolation(self):
        analyzer = self._analyzer()

        with patch.object(
            analyzer,
            "_are_supercat_subcats",
            return_value=([["super", "sub"]], [True]),
        ):
            analyzer.fit_supercat_subcat_pairs(
                self.data,
                pairs_list=[["super", "sub"]],
                isolate_super_subs=False,
            )

        super_meta = analyzer.target_key_feature_meta_vals["super"]
        sub_meta = analyzer.target_key_feature_meta_vals["sub"]
        self.assertEqual(
            analyzer.supercategory_subcategory_pairs,
            [["super", "sub"]],
        )
        self.assertEqual(
            analyzer.reject_null_catcat,
            [["sub", "super", "chi2"]],
        )
        self.assertEqual(
            super_meta["significant_categoric_relationships"],
            ["sub"],
        )
        self.assertEqual(
            sub_meta["significant_categoric_relationships"],
            ["super"],
        )
        self.assertEqual(sub_meta["paired_to_a_supercategory"], ["super"])
        self.assertEqual(
            sub_meta["paired_to_a_supercategory_tests"],
            ["chi2"],
        )
        self.assertEqual(super_meta["paired_to_a_subcategory"], ["sub"])
        self.assertEqual(
            super_meta["paired_to_a_subcategory_tests"],
            ["chi2"],
        )

    def test_successful_isolated_pair_moves_relationship_state(self):
        analyzer = self._analyzer()

        with patch.object(
            analyzer,
            "_are_supercat_subcats",
            return_value=([["super", "sub"]], [True]),
        ):
            analyzer.fit_supercat_subcat_pairs(
                self.data,
                pairs_list=[["super", "sub"]],
                isolate_super_subs=True,
            )

        super_meta = analyzer.target_key_feature_meta_vals["super"]
        sub_meta = analyzer.target_key_feature_meta_vals["sub"]
        self.assertEqual(analyzer.reject_null_catcat, [])
        self.assertEqual(
            super_meta["significant_categoric_relationships"],
            [],
        )
        self.assertEqual(super_meta["significant_categoric_tests"], [])
        self.assertEqual(
            sub_meta["significant_categoric_relationships"],
            [],
        )
        self.assertEqual(sub_meta["significant_categoric_tests"], [])
        self.assertEqual(sub_meta["paired_to_a_supercategory"], ["super"])
        self.assertEqual(
            sub_meta["paired_to_a_supercategory_tests"],
            ["chi2"],
        )
        self.assertEqual(super_meta["paired_to_a_subcategory"], ["sub"])
        self.assertEqual(
            super_meta["paired_to_a_subcategory_tests"],
            ["chi2"],
        )

    def test_successful_target_only_fit_handles_asymmetric_metadata(self):
        analyzer = self._analyzer(targets=("sub",))

        with patch.object(
            analyzer,
            "_are_supercat_subcats",
            return_value=([["super", "sub"]], [True]),
        ):
            analyzer.fit_supercat_subcat_pairs(
                self.data,
                pairs_list=[["super", "sub"]],
                isolate_super_subs=True,
            )

        sub_meta = analyzer.target_key_feature_meta_vals["sub"]
        self.assertEqual(
            analyzer.supercategory_subcategory_pairs,
            [["super", "sub"]],
        )
        self.assertNotIn("super", analyzer.target_key_feature_meta_vals)
        self.assertEqual(sub_meta["paired_to_a_supercategory"], ["super"])
        self.assertEqual(
            sub_meta["paired_to_a_supercategory_tests"],
            ["chi2"],
        )
        self.assertEqual(
            len(sub_meta["paired_to_a_supercategory"]),
            len(sub_meta["paired_to_a_supercategory_tests"]),
        )


if __name__ == "__main__":
    unittest.main()
