"""Free unit tests for calibrated Planner coverage metrics."""

import unittest

from evals.scripts.score_planner_judge_calibration import summarize


class TestPlannerJudgeCalibrationScore(unittest.TestCase):
    def test_reports_both_error_directions_and_unmeasured(self):
        rows = [(True, True), (False, False), (False, True), (True, False), (None, None)]
        sheet = {
            "purpose": "manual_planner_judge_calibration_only",
            "cases": [
                {
                    "aspects": [
                        {"human_covered": human, "judge_covered": judge} for human, judge in rows
                    ]
                }
            ],
        }
        result = summarize(sheet)
        self.assertEqual(result["measured_aspects"], 4)
        self.assertEqual(result["unmeasured_aspects"], 1)
        self.assertEqual(result["agreement"], 0.5)
        self.assertEqual(result["false_covered"], 1)
        self.assertEqual(result["false_missing"], 1)
        self.assertEqual(result["coverage_precision"], 0.5)

    def test_all_unmeasured_cannot_be_mistaken_for_zero(self):
        sheet = {
            "purpose": "manual_planner_judge_calibration_only",
            "cases": [{"aspects": [{"human_covered": None, "judge_covered": None}]}],
        }
        with self.assertRaisesRegex(ValueError, "no_human_and_judge_pairs_measured"):
            summarize(sheet)


if __name__ == "__main__":
    unittest.main()
