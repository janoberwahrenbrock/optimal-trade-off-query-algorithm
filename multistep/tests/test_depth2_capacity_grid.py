from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multistep.scripts.run_depth2_capacity_grid import (
    deterministic_problem,
    load_or_create_checkpoint,
    new_checkpoint,
    update_summary,
    write_checkpoint_atomic,
)


class DepthTwoCapacityGridTest(unittest.TestCase):
    def test_problem_generation_is_independently_reproducible(self) -> None:
        first = deterministic_problem(123, 5, 7, 9)
        repeated = deterministic_problem(123, 5, 7, 9)
        different = deterministic_problem(123, 5, 7, 10)

        self.assertEqual(first[0].entries, repeated[0].entries)
        self.assertEqual(first[1], repeated[1])
        self.assertNotEqual(first[0].entries, different[0].entries)

    def test_atomic_checkpoint_can_be_loaded_for_resume(self) -> None:
        settings = {"planned_problem_count": 1}
        checkpoint = new_checkpoint(settings)
        checkpoint["results"].append(
            {
                "status": "solved",
                "seconds": 1.25,
                "question_count": 3,
            }
        )
        update_summary(checkpoint)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            write_checkpoint_atomic(path, checkpoint)
            loaded = load_or_create_checkpoint(path, settings)

        self.assertEqual(loaded["summary"]["solved_problem_count"], 1)
        self.assertEqual(loaded["summary"]["total_questions_for_solved_problems"], 3)
        self.assertEqual(loaded["run"]["status"], "running")

    def test_resume_rejects_changed_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            write_checkpoint_atomic(path, new_checkpoint({"seed": 1}))

            with self.assertRaisesRegex(RuntimeError, "settings differ"):
                load_or_create_checkpoint(path, {"seed": 2})


if __name__ == "__main__":
    unittest.main()
