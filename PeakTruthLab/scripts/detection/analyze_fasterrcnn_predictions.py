from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from train_fasterrcnn import build_model


def _collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def load_label_map(label_map_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    with label_map_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    labels_obj = obj.get("labels") or {}
    name_to_id = {str(k): int(v) for k, v in labels_obj.items()}
    id_to_name = {int(v): str(k) for k, v in name_to_id.items()}
    return id_to_name, name_to_id


def main() -> int:
    p = argparse.ArgumentParser("Analyze prediction label distribution")
    p.add_argument(
        "--image-root",
        type=str,
        default=r"D:\\LipidBench\\PeakTruthLab\\datasets\\split\\val",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=r"D:\\LipidBench\\PeakTruthLab\\models\\fasterrcnn_mvp_latest.pth",
    )
    p.add_argument(
        "--label-map",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.json"),
    )
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.5",
        help="Comma-separated score thresholds",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images for quick analysis (0 means all)",
    )
    args = p.parse_args()

    image_root = Path(args.image_root).resolve()
    weights = Path(args.weights).resolve()
    label_map_path = Path(args.label_map).resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")

    id_to_name, name_to_id = load_label_map(label_map_path)
    num_classes = int(max(name_to_id.values())) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(num_classes=num_classes)
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    images = _collect_images(image_root)
    if not images:
        raise RuntimeError(f"No images found under: {image_root}")

    if int(args.max_images) > 0:
        images = images[: int(args.max_images)]

    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]

    print("image_root:", image_root)
    print("images:", len(images))
    print("label_map:", label_map_path)
    print("classes:", {k: v for k, v in sorted(id_to_name.items())})

    # Run inference once per image, then reuse across thresholds (CPU-friendly).
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for i, img_path in enumerate(images, start=1):
            if i == 1 or i % 5 == 0 or i == len(images):
                print(f"inference {i}/{len(images)}: {img_path.name}")
            pil = Image.open(img_path).convert("RGB")
            img = to_tensor(pil).to(device)
            out = model([img])[0]
            cached.append((out["scores"].detach().cpu(), out["labels"].detach().cpu()))

    for thr in thresholds:
        counter: Counter[int] = Counter()
        kept_boxes = 0
        for scores, labels in cached:
            keep = (scores >= thr) & (labels > 0)
            labs = labels[keep].tolist()
            kept_boxes += len(labs)
            counter.update(int(x) for x in labs)

        named = {id_to_name.get(k, f"cls_{k}"): v for k, v in sorted(counter.items())}
        print(f"\nthreshold={thr:.2f}")
        print("kept_boxes:", kept_boxes)
        print("label_counts:", named)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
