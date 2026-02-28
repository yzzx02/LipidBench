from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes

from train_model import build_model


def load_label_map(label_map_path: Path) -> tuple[dict[int, str], dict[str, int], dict[str, str]]:
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map not found: {label_map_path}")
    with label_map_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    labels_obj = obj.get("labels") or {}
    aliases_obj = obj.get("aliases") or {}
    name_to_id = {str(k): int(v) for k, v in labels_obj.items()}
    id_to_name = {int(v): str(k) for k, v in name_to_id.items()}
    aliases = {str(k): str(v) for k, v in aliases_obj.items()}
    return id_to_name, name_to_id, aliases


def _collect_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_root = Path(args.image_root).resolve()
    weights = Path(args.weights).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    id_to_name, name_to_id, _aliases = load_label_map(Path(args.label_map))

    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "label_map" in state and args.label_map is None:
        # kept for backward compatibility; prefer explicit --label-map
        name_to_id = {str(k): int(v) for k, v in (state.get("label_map") or {}).items()}
        id_to_name = {int(v): str(k) for k, v in name_to_id.items()}

    num_classes = int(max(name_to_id.values())) + 1
    model = build_model(num_classes=num_classes)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    images = _collect_images(image_root)
    if not images:
        raise RuntimeError(f"No images found under: {image_root}")

    with torch.no_grad():
        for img_path in images:
            pil = Image.open(img_path).convert("RGB")
            img_float = to_tensor(pil).to(device)  # [0,1]

            outputs = model([img_float])[0]
            boxes = outputs["boxes"].detach().cpu()
            scores = outputs["scores"].detach().cpu()
            labels = outputs["labels"].detach().cpu()

            keep = (scores >= float(args.score_threshold)) & (labels > 0)
            boxes_kept = boxes[keep]
            scores_kept = scores[keep]
            labels_kept = labels[keep]

            # draw_bounding_boxes expects uint8 C,H,W
            img_uint8 = (img_float.detach().cpu() * 255.0).clamp(0, 255).to(torch.uint8)

            if boxes_kept.numel() > 0:
                text_labels = []
                for lab, sc in zip(labels_kept.tolist(), scores_kept.tolist()):
                    name = id_to_name.get(int(lab), f"cls_{int(lab)}")
                    text_labels.append(f"{name} {float(sc):.2f}")
                drawn = draw_bounding_boxes(
                    img_uint8,
                    boxes=boxes_kept,
                    labels=text_labels,
                    colors="red",
                    width=2,
                )
            else:
                drawn = img_uint8

            out_img = to_pil_image(drawn)
            rel = img_path.relative_to(image_root)
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            out_img.save(dst)

    print(f"image_root: {image_root}")
    print(f"weights:    {weights}")
    print(f"out_root:   {out_root}")
    print(f"processed:  {len(images)}")
    print(f"score_th:   {args.score_threshold}")


def parse_args():
    parser = argparse.ArgumentParser("Minimal inference visualization for Faster R-CNN MVP")
    parser.add_argument(
        "--image-root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\datasets\split\val",
        help="Validation (or new) image directory",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\models\fasterrcnn_mvp.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"D:\LipidBench\PeakTruthLab\outputs",
        help="Directory to save visualized predictions",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "label_map.json"),
        help="JSON file defining label->class_id mapping",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
