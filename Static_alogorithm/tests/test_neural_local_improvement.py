import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


STATIC_DIR = Path(__file__).resolve().parents[1]
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

from GA.GA import Individual  # noqa: E402
import benchmark_neural_local_improvement as study  # noqa: E402
import neural_local_improvement as improvement  # noqa: E402


class NeuralLocalImprovementTests(unittest.TestCase):
    def make_individual(self):
        return Individual(order=["pickup", "unload"], amr_assignment=["AMR1"])

    def test_zero_budgets_clone_without_search(self):
        original = self.make_individual()
        with mock.patch.object(improvement, "local_improve") as local_search:
            result = improvement.apply_neural_local_improvement(original, [], 0, 0)

        local_search.assert_not_called()
        self.assertIsNot(result.individual, original)
        self.assertIsNot(result.individual.order, original.order)
        self.assertIsNot(result.individual.amr_assignment, original.amr_assignment)
        self.assertEqual(result.postprocess_time, 0.0)

    def test_stage_order_collision_flags_and_timing(self):
        original = self.make_individual()
        with mock.patch.object(improvement, "local_improve", side_effect=lambda individual, *args, **kwargs: individual) as local_search:
            with mock.patch.object(improvement.time, "perf_counter", side_effect=[1.0, 2.5, 5.0, 9.0]):
                result = improvement.apply_neural_local_improvement(original, [], 3, 4)

        self.assertEqual(local_search.call_count, 2)
        self.assertFalse(local_search.call_args_list[0].kwargs["check_collision"])
        self.assertEqual(local_search.call_args_list[0].kwargs["max_iters"], 3)
        self.assertTrue(local_search.call_args_list[1].kwargs["check_collision"])
        self.assertEqual(local_search.call_args_list[1].kwargs["max_iters"], 4)
        self.assertEqual(result.simplified_time, 1.5)
        self.assertEqual(result.collision_time, 4.0)
        self.assertEqual(result.postprocess_time, 5.5)

    def test_explicit_stage_seeds_are_repeatable_and_isolated(self):
        observed = []

        def randomized_search(individual, *args, **kwargs):
            observed.append(random.random())
            return individual

        random.seed(99)
        expected_next = random.random()
        random.seed(99)
        with mock.patch.object(improvement, "local_improve", side_effect=randomized_search):
            improvement.apply_neural_local_improvement(
                self.make_individual(), [], 1, 1, simplified_seed=11, collision_seed=22
            )
            first = list(observed)
            observed.clear()
            improvement.apply_neural_local_improvement(
                self.make_individual(), [], 1, 1, simplified_seed=11, collision_seed=22
            )
            second = list(observed)

        self.assertEqual(first, second)
        self.assertEqual(random.random(), expected_next)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            improvement.apply_neural_local_improvement(self.make_individual(), [], -1, 0)


class StudyLogicTests(unittest.TestCase):
    def result_row(self, *, config=(0, 0), valid=True, makespan=100.0, total=1.0, repeat=0):
        return {
            "phase": "tune",
            "model": "gnn",
            "job_count": "20",
            "sample_index": "0",
            "repeat": str(repeat),
            "simplified_iters": str(config[0]),
            "collision_iters": str(config[1]),
            "status": "ok" if valid else "invalid",
            "valid": "true" if valid else "false",
            "makespan": str(makespan),
            "invalid_jobs": "0" if valid else "1",
            "inference_time": "0.5",
            "simplified_time": "0.2",
            "collision_time": "0.3",
            "postprocess_time": "0.5",
            "total_time": str(total),
        }

    def summary_row(self, config, quality, runtime, validity=1.0):
        return {
            "phase": "tune",
            "model": "gnn",
            "job_count": "overall",
            "simplified_iters": config[0],
            "collision_iters": config[1],
            "valid_rate": validity,
            "mean_paired_makespan_ratio": quality,
            "mean_total_time": runtime,
        }

    def test_grid_expansion_and_stable_seed(self):
        self.assertEqual(study.parse_nonnegative_grid("0, 10,10,25"), [0, 10, 25])
        self.assertEqual(study.expand_grid([0, 100], [0, 10]), [(0, 0), (0, 10), (100, 0), (100, 10)])
        self.assertEqual(study.stable_seed(42, "a", 1), study.stable_seed(42, "a", 1))
        self.assertNotEqual(study.stable_seed(42, "a", 1), study.stable_seed(42, "a", 2))

    def test_detail_writer_resume_key_skips_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.csv"
            row = self.result_row()
            row.update({"simplified_seed": 1, "collision_seed": 2, "error": ""})
            writer = study.DetailWriter(path, resume=False)
            self.assertTrue(writer.append(row))
            self.assertFalse(writer.append(row))
            writer.close()
            writer = study.DetailWriter(path, resume=True)
            self.assertFalse(writer.append(row))
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_aggregation_computes_paired_ratio_and_overall(self):
        rows = [
            self.result_row(config=(0, 0), makespan=100, total=1),
            self.result_row(config=(100, 0), makespan=90, total=2),
        ]
        summary = study.aggregate_rows(rows)
        improved = next(
            row for row in summary
            if row["job_count"] == 20 and study.config_tuple(row) == (100, 0)
        )
        self.assertAlmostEqual(improved["mean_paired_makespan_ratio"], 0.9)
        self.assertTrue(any(row["job_count"] == "overall" for row in summary))

    def test_validity_guard_excludes_faster_invalid_configuration(self):
        rows = [
            self.summary_row((0, 0), 1.0, 1.0, 1.0),
            self.summary_row((100, 0), 0.8, 1.1, 0.9),
            self.summary_row((0, 10), 0.95, 1.2, 1.0),
        ]
        eligible, frontier, baseline_validity = study.eligible_and_frontier(rows)
        self.assertEqual(baseline_validity, 1.0)
        self.assertNotIn((100, 0), [study.config_tuple(row) for row in eligible])
        self.assertIn((0, 10), [study.config_tuple(row) for row in frontier])

    def test_knee_tie_breaks_on_fewer_iterations(self):
        tied = [
            self.summary_row((100, 0), 0.9, 2.0),
            self.summary_row((200, 0), 0.9, 2.0),
        ]
        self.assertEqual(study.config_tuple(study.select_knee(tied)), (100, 0))

    def test_shortlist_contains_baseline_and_at_most_four_configs(self):
        rows = [
            self.summary_row((0, 0), 1.0, 1.0),
            self.summary_row((100, 0), 0.96, 1.2),
            self.summary_row((250, 0), 0.92, 1.8),
            self.summary_row((500, 10), 0.90, 3.0),
        ]
        selected = study.select_shortlist(rows)
        self.assertEqual(selected[0], (0, 0))
        self.assertLessEqual(len(selected), 4)

    def test_sweep_cli_defaults(self):
        args = study.prepare_args(study.build_parser().parse_args([]))
        self.assertEqual(args.simplified_grid_values, [0, 100, 250, 500, 1000, 2000])
        self.assertEqual(args.collision_grid_values, [0, 10, 25, 50, 100, 200])
        self.assertEqual(args.search_repeats, 3)


if __name__ == "__main__":
    unittest.main()
