"""Run DETR on two sample images and save detections."""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime import require_cuda, write_result  # noqa: E402


def main():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/detection")
    args = parser.parse_args()
    device = require_cuda()
    model_id = "facebook/detr-resnet-50"
    revision = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = (
        AutoModelForObjectDetection.from_pretrained(model_id, revision=revision).to(device).eval()
    )
    images = []
    for name in ("000000039769", "000000004016"):
        with urlopen(
            f"https://raw.githubusercontent.com/huggingface/transformers/"
            f"1bd535f3c34db453de91433f1c49c20899584d92/tests/fixtures/tests_samples/COCO/{name}.png",
            timeout=60,
        ) as response:
            images.append(Image.open(response).convert("RGB"))
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        predictions = model(**inputs)
    sizes = torch.tensor([(image.height, image.width) for image in images], device=device)
    detections = processor.post_process_object_detection(
        predictions, threshold=0.7, target_sizes=sizes
    )
    rows = []
    for index, detection in enumerate(detections):
        if not torch.isfinite(detection["boxes"]).all():
            raise RuntimeError("DETR produced non-finite boxes")
        for score, label, box in zip(
            detection["scores"], detection["labels"], detection["boxes"], strict=True
        ):
            rows.append(
                {
                    "image": index,
                    "label": model.config.id2label[label.item()],
                    "score": score.item(),
                    "box": box.tolist(),
                }
            )
    if not rows:
        raise RuntimeError("No objects detected in the sample images")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "detections.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_result(
        output,
        {"images": len(images), "detections": len(rows), "model": model_id, "revision": revision},
    )


if __name__ == "__main__":
    main()
