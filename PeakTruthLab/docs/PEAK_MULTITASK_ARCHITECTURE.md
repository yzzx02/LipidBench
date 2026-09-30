# PeakMultiTaskRCNN Architecture

## Scope

`PeakMultiTaskRCNN` is ChromaPeak's joint detector and candidate validator. This model module defines the shared architecture, loss interface, and configuration; dataset loading, training, evaluation, and signal processing live in separate modules. The maintained joint training and locked evaluation workflows are implemented under `PeakTruthLab/scripts/detection/`.

Whole-image candidate-classification comparisons remain under `PeakTruthLab/scripts/convnext/`. Training entry points are limited to the current RTX workflows. The released joint model uses 16 attributes and naive concatenation.

## Input size convention

The official YAML configuration uses `image_min_size: 480` and `image_max_size: 480`, matching the current 480×480 EIC image convention. A 480×480 image therefore remains 480×480 through `GeneralizedRCNNTransform`; detection boxes and Candidate boxes do not undergo unnecessary coordinate scaling.

The previous default downscale to 256 is not used because it would discard spatial detail and alter box coordinates before the model sees the project-standard image. The 480 setting is a current data convention, not a permanently hardcoded network limitation. Constructor/config overrides still permit other image sizes, and the random CPU Smoke Test intentionally overrides the size to 96.

## Shared feature extractor

`ConvNeXtTinyFPNBackbone` exposes TorchVision ConvNeXt-Tiny stages as C2, C3, C4, and C5 with 96, 192, 384, and 768 channels. `FeaturePyramidNetwork` maps exactly these four stages to 256-channel P2-P5 maps:

- C2 (96 channels) -> P2 (256 channels)
- C3 (192 channels) -> P3 (256 channels)
- C4 (384 channels) -> P4 (256 channels)
- C5 (768 channels) -> P5 (256 channels)

Both tasks reuse exactly one ConvNeXt-Tiny + FPN forward pass:

```text
EIC image
-> ConvNeXt-Tiny C2/C3/C4/C5
-> FPN P2/P3/P4/P5
   |-> Faster R-CNN full-window detection
   `-> Candidate RoI pooling and candidate validation
```

RPN receives all four FPN maps. Detection `MultiScaleRoIAlign` and Candidate `MultiScaleRoIAlign` both use `p2`, `p3`, `p4`, and `p5`. No additional pyramid level is generated.

TorchVision defines anchor aspect ratio as height divided by width. Its implementation computes `height = sqrt(ratio) * scale` and `width = scale / sqrt(ratio)`, so ratios 2, 4, 8, and 16 represent progressively taller, narrower candidate boxes. The four-level YAML Anchor values are an initial architecture configuration, not tuned experimental results; later adjustment must use training-set annotation statistics without adding pyramid levels.

## Task 1: full-window True_Peak detection

The detection branch is a standard Faster R-CNN pipeline with RPN, `MultiScaleRoIAlign`, foreground/background classification, and box regression. The class count is fixed to two: implicit background class 0 and `True_Peak` class 1. Empty, single-box, and multi-box images are valid.

The detection branch is image-only. Candidate-specific attributes are undefined before an unknown full-window peak is localized, and copying the original candidate's attributes to unrelated proposals would be semantically invalid.

TorchVision supplies the unmodified detection losses:

- `loss_objectness`
- `loss_rpn_box_reg`
- `loss_classifier`
- `loss_box_reg`

## Task 2: original candidate validation

Each image supplies one original `seed_box` and one attribute vector. The Candidate box is resized with the same independent width and height scale factors used by `GeneralizedRCNNTransform`. `MultiScaleRoIAlign` pools the candidate's visual feature from the already-computed shared FPN maps, so ConvNeXt and FPN are not run again.

The model uses exactly 16 attributes: SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE. The configuration, preprocessing and model input must follow this dimension and order. The Candidate head supports:

- `image_only`
- `attr_only`
- `naive_concat`
- `gated_fusion`

Both the maintained YAML and Python constructors default to `naive_concat`, matching the released model.

The Candidate label is independent of full-window detection labels. An image may contain a true peak far from a false original Candidate: the detector can localize the true peak while the Candidate head rejects the original candidate.

The Candidate branch adds `loss_seed_cls`, multiplied by `seed_loss_weight`. Supported losses are BCE, weighted BCE, and focal loss. `future_giou_enabled` remains a documented placeholder; current RPN and RoI box losses are unchanged.

## Why Faster R-CNN rather than DETR

Faster R-CNN directly supports zero, one, or multiple boxes and provides the required proposal, RoI pooling, classification, and regression losses. The Candidate branch can naturally reuse FPN RoI features. DETR and Hungarian matching would add a different proposal and loss framework without being required by the current architecture.

## Forward interfaces

Public descriptions use Candidate (候选峰). Historical `seed_box`, `seed_boxes`, `seed_labels`, `seed_logits`, `seed_probabilities`, and `loss_seed_cls` keys remain unchanged for script and checkpoint compatibility.

Training returns:

```python
{
    "loss_classifier": Tensor,
    "loss_box_reg": Tensor,
    "loss_objectness": Tensor,
    "loss_rpn_box_reg": Tensor,
    "loss_seed_cls": Tensor,
}
```

Inference returns:

```python
{
    "detections": list[dict],
    "seed_logits": Tensor[B],
    "seed_probabilities": Tensor[B],
}
```

## Implementation boundaries

Manifest/Dataset interfaces, train-only attribute preprocessing, real training, locked evaluation, EIC extraction, RT-boundary refinement, and signal integration utilities are implemented in their respective modules and scripts. These are not operations performed by the neural network itself. A single packaged raw-mzML-to-validated-quantification command is not yet provided.

The current detector uses standard TorchVision RPN/RoI losses and NMS. Soft-NMS, GIoU replacement heads, DETR/Hungarian matching, and an automated detection-to-RT-to-recomputed-attributes second stage are not part of the released model.
