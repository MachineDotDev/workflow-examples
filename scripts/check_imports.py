"""Verify the locked ML libraries import on Linux without provisioning a GPU."""

from datasets import load_dataset
from peft import LoraConfig
from torchvision.models import resnet50
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoModelForObjectDetection
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

assert all(
    (
        load_dataset,
        AutoModelForCausalLM,
        LoraConfig,
        resnet50,
        AutoImageProcessor,
        AutoModelForObjectDetection,
        GRPOTrainer,
        SFTTrainer,
    )
)
SFTConfig(
    output_dir="/tmp/sft-import-check",
    use_cpu=True,
    bf16=False,
    max_length=128,
    dataset_text_field="text",
    report_to="none",
)
GRPOConfig(
    output_dir="/tmp/grpo-import-check",
    use_cpu=True,
    bf16=False,
    per_device_train_batch_size=2,
    num_generations=2,
    generation_batch_size=2,
    max_completion_length=32,
    report_to="none",
)
print("Locked ML imports and trainer configuration passed")
