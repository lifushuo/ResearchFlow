"""Offline, zero-model tests for the fixed Planner report A/B split."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.report_ab_dataset import (
    EXCLUDED_CASE_ID,
    INDEPENDENT_REPORT_SET,
    ORIGINAL_REPORT_SET,
    fixed_report_queries,
    fixed_set_hash,
    selected_report_dataset_for_experiment,
)


class TestPlannerReportABDataset(unittest.TestCase):
    def test_nine_rows_preserve_exact_original_fields_and_order(self):
        original = fixed_report_queries(ORIGINAL_REPORT_SET)
        independent = fixed_report_queries(INDEPENDENT_REPORT_SET)
        self.assertEqual(len(original), 10)
        self.assertEqual(len(independent), 9)
        self.assertEqual(independent, [row for row in original if row["id"] != EXCLUDED_CASE_ID])
        self.assertNotIn(EXCLUDED_CASE_ID, [row["id"] for row in independent])

    def test_both_old_and_new_experiments_resolve_by_frozen_hash(self):
        for path in (ORIGINAL_REPORT_SET, INDEPENDENT_REPORT_SET):
            rows = fixed_report_queries(path)
            restored_path, restored_rows = selected_report_dataset_for_experiment(
                {"report_dataset_sha256": fixed_set_hash(rows)}
            )
            self.assertEqual(restored_path, path)
            self.assertEqual(restored_rows, rows)
        with self.assertRaisesRegex(ValueError, "not_recognized"):
            selected_report_dataset_for_experiment({"report_dataset_sha256": "not-a-real-hash"})

    def test_modified_or_reordered_benchmark_is_rejected(self):
        raw = json.loads(INDEPENDENT_REPORT_SET.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            target = Path(directory) / "custom.json"
            mutated = json.loads(json.dumps(raw))
            mutated["queries"][0]["query"] += " extra instruction"
            target.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not_an_approved_fixed_set"):
                fixed_report_queries(target)
            reordered = json.loads(json.dumps(raw))
            reordered["queries"].reverse()
            target.write_text(json.dumps(reordered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not_an_approved_fixed_set"):
                fixed_report_queries(target)


if __name__ == "__main__":
    unittest.main()
