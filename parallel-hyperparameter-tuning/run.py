"""Train ResNet on CIFAR-10 with one matrix configuration."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime import require_cuda, write_result  # noqa: E402


def main():
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/tuning")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or args.batch_size not in (8, 16):
        parser.error("steps must be 1..1000 and batch size must be 8 or 16")
    device = require_cuda()
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    data = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    loader = DataLoader(
        Subset(data, range(min(len(data), args.steps * args.batch_size))),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = models.resnet50(weights=None, num_classes=10).to(device)
    before = model.fc.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if len(losses) == args.steps:
            break
    if torch.equal(before, model.fc.weight):
        raise RuntimeError("Training did not change model parameters")
    write_result(
        args.output,
        {
            "steps": len(losses),
            "losses": losses,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
    )


if __name__ == "__main__":
    main()
