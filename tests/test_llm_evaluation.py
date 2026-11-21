"""CPU tests for the LLM evaluation gate and answerer (no model download)."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "llm-evaluation"))
import answerer  # noqa: E402
from gate import regression_test  # noqa: E402


class GateTest(unittest.TestCase):
    def test_identical_answers_pass(self):
        result = regression_test([True, False, True], [True, False, True])
        self.assertEqual((result["lost"], result["gained"], result["p_value"]), (0, 0, 1.0))
        self.assertFalse(result["regression"])

    def test_one_sided_exact_p_value(self):
        # 9 lost, 1 gained: P(X >= 9 | n=10, p=0.5) = 11/1024
        baseline = [True] * 9 + [False] + [True] * 5
        candidate = [False] * 9 + [True] + [True] * 5
        result = regression_test(baseline, candidate)
        self.assertAlmostEqual(result["p_value"], 11 / 1024)
        self.assertTrue(result["regression"])

    def test_improvement_is_not_a_regression(self):
        result = regression_test([False] * 10, [True] * 10)
        self.assertEqual(result["gained"], 10)
        self.assertFalse(result["regression"])

    def test_small_loss_is_not_significant(self):
        result = regression_test([True, True, False], [False, True, False])
        self.assertAlmostEqual(result["p_value"], 0.5)
        self.assertFalse(result["regression"])

    def test_rejects_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            regression_test([True], [True, False])
        with self.assertRaises(ValueError):
            regression_test([], [])


class StubTokenizer:
    """One token per character."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % 128 for c in text])


class StubModel:
    """Predicts a fixed next-token distribution that strongly prefers one token."""

    device = torch.device("cpu")

    def __init__(self, preferred):
        self.preferred = ord(preferred) % 128

    def __call__(self, input_ids):
        logits = torch.zeros(1, input_ids.shape[1], 128)
        logits[..., self.preferred] = 10.0
        return SimpleNamespace(logits=logits)


class AnswererTest(unittest.TestCase):
    def test_prompt_keeps_whole_question(self):
        question = "Which of these would help to prevent infections from spreading? " * 2
        self.assertEqual(answerer.build_prompt(question), f"Question: {question.strip()}\nAnswer:")

    def test_choose_picks_highest_mean_logprob(self):
        model = StubModel("z")
        choices = ["abc", "zzz", "zz a"]
        self.assertEqual(answerer.choose(model, StubTokenizer(), "q", choices), 1)

    def test_mean_not_sum_so_short_choices_are_not_favoured(self):
        # Both pay one unlikely leading-space token; a summed score would pick
        # the shorter choice, the mean picks the one that is mostly likely tokens.
        model = StubModel("z")
        self.assertEqual(answerer.choose(model, StubTokenizer(), "q", ["zzzzzzzz", "z"]), 0)


def load_run_module():
    spec = importlib.util.spec_from_file_location(
        "llm_eval_run", root / "llm-evaluation" / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoadAnswererTest(unittest.TestCase):
    def test_each_version_imports_its_own_helpers(self):
        run = load_run_module()
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as cand:
            for directory, value in ((base, "baseline"), (cand, "candidate")):
                Path(directory, "prompts.py").write_text(f"PREFIX = {value!r}\n")
                Path(directory, "answerer.py").write_text(
                    "from prompts import PREFIX\n\n\ndef choose(*args):\n    return PREFIX\n"
                )
            baseline = run.load_answerer(base, "answerer_baseline_test")
            candidate = run.load_answerer(cand, "answerer_candidate_test")
        self.assertEqual(baseline.choose(), "baseline")
        self.assertEqual(candidate.choose(), "candidate")


class DefectPatchTest(unittest.TestCase):
    def test_defect_patches_apply_to_the_current_answerer(self):
        for patch in (root / "llm-evaluation" / "defects").glob("*.patch"):
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)


if __name__ == "__main__":
    unittest.main()
