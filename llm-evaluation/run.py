"""Gate a change to the multiple-choice answerer against a baseline on ARC Easy."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "scripts"))
sys.path.insert(0, str(here))
from gate import regression_test  # noqa: E402
from runtime import require_cuda, write_result  # noqa: E402

MODEL = ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")
DATASET = ("allenai/ai2_arc", "ARC-Easy", "210d026faf9955653af8916fad021475a3f00453")


def load_answerer(directory, name):
    """Import one version's answerer.py, resolving its own helper modules.

    The baseline and candidate may both import helpers with the same name (for
    example `prompts`). Put this version's directory first on sys.path while it
    loads, then forget any module imported from that directory, so the other
    version imports its own copy instead of reusing this one.
    """
    directory = Path(directory).resolve()
    path = directory / "answerer.py"
    if not path.is_file():
        raise FileNotFoundError(f"No answerer.py in {directory}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
        for loaded in set(sys.modules) - before:
            origin = getattr(sys.modules[loaded], "__file__", None)
            if origin and Path(origin).resolve().is_relative_to(directory):
                del sys.modules[loaded]
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Directory with the baseline answerer.py")
    parser.add_argument("--candidate", default=str(here), help="Directory with the candidate")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()
    if not 1 <= args.limit <= 2376:
        parser.error("limit must be 1..2376 (the ARC Easy test split)")
    if not 0 < args.alpha < 1:
        parser.error("alpha must be between 0 and 1")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = require_cuda()
    torch.use_deterministic_algorithms(True)
    model_id, model_revision = MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, dtype=torch.float16
    ).to(device)
    model.eval()
    dataset_id, config, dataset_revision = DATASET
    questions = load_dataset(dataset_id, config, split="test", revision=dataset_revision)
    questions = questions.select(range(args.limit))

    answers = [row["choices"]["label"].index(row["answerKey"]) for row in questions]
    correct = {}
    predictions = {}
    for name, directory in (("baseline", args.baseline), ("candidate", args.candidate)):
        answerer = load_answerer(directory, f"answerer_{name}")
        predictions[name] = [
            answerer.choose(model, tokenizer, row["question"], row["choices"]["text"])
            for row in questions
        ]
        correct[name] = [p == a for p, a in zip(predictions[name], answers, strict=True)]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w") as file:
        for row, answer, base, cand in zip(
            questions, answers, predictions["baseline"], predictions["candidate"], strict=True
        ):
            record = {"id": row["id"], "answer": answer, "baseline": base, "candidate": cand}
            file.write(json.dumps(record) + "\n")

    decision = regression_test(correct["baseline"], correct["candidate"], args.alpha)
    write_result(
        args.output,
        {
            "model": {"id": model_id, "revision": model_revision, "dtype": "float16"},
            "dataset": {"id": dataset_id, "config": config, "revision": dataset_revision},
            "split": "test",
            "limit": args.limit,
            "test": "one-sided exact McNemar",
            **decision,
        },
    )
    if decision["regression"]:
        print(
            f"::error::Candidate regressed: lost {decision['lost']} questions, "
            f"gained {decision['gained']} (p={decision['p_value']:.2g} < {args.alpha})"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
