"""Free tests for the manual Planner Judge calibration sheet generator."""

import unittest

from evals.scripts.make_planner_judge_calibration_sheet import make_sheet


class TestPlannerJudgeCalibrationSheet(unittest.TestCase):
    def test_only_train_with_blank_labels_and_backflow_first(self):
        data = {
            "cases": [
                {
                    "split": "holdout",
                    "case_id": "secret",
                    "query": "not for training",
                    "required_aspects": ["a"],
                },
                {
                    "split": "train",
                    "case_id": "seed",
                    "query": "seed task",
                    "required_aspects": ["a"],
                },
                {
                    "split": "train",
                    "case_id": "real",
                    "query": "real task",
                    "required_aspects": ["a", "b"],
                    "source_bad_case_id": "bc-1",
                },
                {
                    "split": "validation",
                    "case_id": "val",
                    "query": "validation task",
                    "required_aspects": ["a"],
                },
            ]
        }
        output = make_sheet(data, max_cases=2)
        self.assertEqual([x["case_id"] for x in output["cases"]], ["real", "seed"])
        self.assertNotIn("not for training", str(output))
        self.assertNotIn("validation task", str(output))
        self.assertIsNone(output["cases"][0]["aspects"][0]["human_covered"])
        self.assertIsNone(output["cases"][0]["aspects"][0]["judge_covered"])
        self.assertEqual(output["purpose"], "manual_planner_judge_calibration_only")

    def test_cannot_turn_empty_or_invalid_cases_into_labels(self):
        with self.assertRaisesRegex(ValueError, "no_train_cases"):
            make_sheet({"cases": [{"split": "holdout", "case_id": "h"}]}, max_cases=1)
        with self.assertRaisesRegex(ValueError, "invalid_train_case"):
            make_sheet(
                {
                    "cases": [
                        {"split": "train", "case_id": "a", "query": "q", "required_aspects": []}
                    ]
                },
                max_cases=1,
            )

    def test_evaluation_case_backflow_without_bad_case_id_is_prioritized(self):
        """评测案例通道的回流没有 source_bad_case_id，仍必须优先进入校准表。"""
        data = {
            "cases": [
                {"split": "train", "case_id": "seed-1", "query": "q1", "required_aspects": ["a"]},
                {
                    "split": "train",
                    "case_id": "planner-review-llm-memory-ca4295f7",
                    "query": "q2",
                    "required_aspects": ["a", "b"],
                },
            ]
        }
        output = make_sheet(data, max_cases=2)
        self.assertEqual(output["cases"][0]["case_id"], "planner-review-llm-memory-ca4295f7")


if __name__ == "__main__":
    unittest.main()
