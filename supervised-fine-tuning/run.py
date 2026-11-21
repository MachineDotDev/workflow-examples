"""Train a small LoRA adapter, with optional checkpoint resume."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime import require_cuda, write_result  # noqa: E402


def adapter_changed(model, before, checkpoint=None):
    """Whether training changed the trainable weights.

    A fresh run compares with the weights before train(). A resumed run compares
    with the adapter saved in the checkpoint, because train() loads those weights
    after `before` was taken.
    """
    import torch

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if checkpoint is None:
        reference = before
    else:
        from safetensors.torch import load_file

        saved = load_file(str(Path(checkpoint) / "adapter_model.safetensors"))
        reference = {}
        for name in trainable:
            key = name.replace(".default.", ".")
            if key in saved:
                reference[name] = saved[key]
        if not reference:
            raise RuntimeError("Checkpoint adapter weights don't match the model")
    compared = [n for n in trainable if n in reference]
    return any(
        not torch.equal(reference[n].to(trainable[n].device, trainable[n].dtype), trainable[n])
        for n in compared
    )


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default="outputs/sft")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be 1..1000")
    require_cuda()
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    dataset = Dataset.from_dict(
        {
            "text": [
                "Machine runs GPU jobs from GitHub Actions. Choose a GPU with the runs-on label.",
                "Save training checkpoints so a later job can resume from the saved state.",
                "A smoke run verifies training works. Use a held-out dataset to measure quality.",
                "Keep credentials out of source control. GitHub supplies a scoped job token.",
            ]
            * 8
        }
    )
    config = SFTConfig(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_length=128,
        dataset_text_field="text",
        bf16=False,
        fp16=False,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        logging_steps=1,
        # Constant, so the resumed step still trains: a decaying schedule sized
        # to the first run reaches zero at its last step.
        lr_scheduler_type="constant",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )
    before = {
        name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad
    }
    result = trainer.train(resume_from_checkpoint=args.resume)
    changed = adapter_changed(trainer.model, before, args.resume)
    if not changed or trainer.state.global_step != args.steps:
        raise RuntimeError("Training failed to update adapter or reach requested step")
    trainer.save_model(str(Path(args.output) / "adapter"))
    tokenizer.save_pretrained(str(Path(args.output) / "adapter"))
    write_result(
        args.output,
        {
            "steps": trainer.state.global_step,
            "loss": result.training_loss,
            "resumed": bool(args.resume),
            "model": model_id,
            "revision": revision,
        },
    )


if __name__ == "__main__":
    main()
