from __future__ import annotations

import pytest
import torch

from lipidbench.models import GatedSeedFusionHead, PeakMultiTaskRCNN, resize_seed_boxes


def _target(boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    if boxes:
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
        }
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


def _small_model(attr_dim: int = 13) -> PeakMultiTaskRCNN:
    return PeakMultiTaskRCNN(
        attr_dim=attr_dim,
        pretrained=False,
        anchor_sizes=((8,), (16,), (32,), (64,)),
        anchor_aspect_ratios=tuple((1.0, 2.0, 4.0) for _ in range(4)),
        image_min_size=64,
        image_max_size=96,
        rpn_pre_nms_top_n_train=32,
        rpn_pre_nms_top_n_test=16,
        rpn_post_nms_top_n_train=16,
        rpn_post_nms_top_n_test=8,
        rpn_batch_size_per_image=32,
        box_score_thresh=0.0,
        box_detections_per_img=5,
        box_batch_size_per_image=16,
    )


def test_seed_box_resize_uses_independent_width_and_height_scales() -> None:
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    resized = resize_seed_boxes(boxes, original_size=(100, 200), new_size=(200, 100))
    expected = torch.tensor([[5.0, 40.0, 25.0, 120.0]])
    torch.testing.assert_close(resized, expected)


def test_official_480_transform_preserves_image_and_box_coordinates() -> None:
    model = PeakMultiTaskRCNN(
        pretrained=False,
        anchor_sizes=((16,), (32,), (64,), (128,)),
        anchor_aspect_ratios=tuple((1.0, 2.0, 4.0, 8.0, 16.0) for _ in range(4)),
    )
    image = torch.rand(3, 480, 480)
    target = _target([[120.0, 30.0, 200.0, 450.0]])
    seed_box = torch.tensor([[125.0, 35.0, 195.0, 445.0]])

    image_list, transformed_targets = model.detector.transform([image], [target])

    assert image_list.image_sizes == [(480, 480)]
    assert image_list.tensors.shape[-2:] == (480, 480)
    assert transformed_targets is not None
    torch.testing.assert_close(transformed_targets[0]["boxes"], target["boxes"])
    resized_seed_box = resize_seed_boxes(
        seed_box,
        original_size=(480, 480),
        new_size=image_list.image_sizes[0],
    )
    torch.testing.assert_close(resized_seed_box, seed_box)


@pytest.mark.parametrize("attr_dim", [13, 15])
def test_seed_fusion_supports_13_and_15_attributes(attr_dim: int) -> None:
    head = GatedSeedFusionHead(attr_dim=attr_dim, mode="gated_fusion")
    roi_features = torch.randn(2, 256, 7, 7, requires_grad=True)
    attributes = torch.randn(2, attr_dim)
    logits = head(roi_features, attributes)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert roi_features.grad is not None


def test_multitask_model_accepts_any_positive_attribute_dimension() -> None:
    model = _small_model(attr_dim=7)
    assert model.attr_dim == 7

    with pytest.raises(ValueError, match="positive integer"):
        _small_model(attr_dim=0)


def test_multitask_model_requires_four_anchor_levels() -> None:
    with pytest.raises(
        ValueError,
        match=r"anchor_sizes and anchor_aspect_ratios must each define P2-P5 \(4 levels\)",
    ):
        PeakMultiTaskRCNN(
            pretrained=False,
            anchor_sizes=((8,), (16,), (32,)),
            anchor_aspect_ratios=tuple((1.0, 2.0, 4.0) for _ in range(4)),
        )


@pytest.mark.parametrize("mode", ["image_only", "attr_only", "naive_concat", "gated_fusion"])
def test_seed_fusion_modes_return_one_logit_per_seed(mode: str) -> None:
    head = GatedSeedFusionHead(attr_dim=13, mode=mode)
    logits = head(torch.randn(3, 256, 7, 7), torch.randn(3, 13))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_multitask_train_backward_and_inference_share_one_backbone_forward() -> None:
    torch.manual_seed(7)
    model = _small_model(attr_dim=13)
    images = [torch.rand(3, 64, 64) for _ in range(4)]
    targets = [
        _target([]),
        _target([[20.0, 5.0, 36.0, 58.0]]),
        _target([[6.0, 4.0, 18.0, 60.0], [42.0, 8.0, 56.0, 58.0]]),
        # A true detection target is far from the false seed below.
        _target([[44.0, 6.0, 58.0, 59.0]]),
    ]
    seed_boxes = [
        torch.tensor([[22.0, 8.0, 36.0, 56.0]]),
        torch.tensor([[20.0, 5.0, 36.0, 58.0]]),
        torch.tensor([[6.0, 4.0, 18.0, 60.0]]),
        torch.tensor([[4.0, 8.0, 16.0, 56.0]]),
    ]
    attributes = torch.randn(4, 13)
    seed_labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

    expected_feature_names = ("p2", "p3", "p4", "p5")
    assert tuple(model.detector.roi_heads.box_roi_pool.featmap_names) == expected_feature_names
    assert tuple(model.seed_roi_pool.featmap_names) == expected_feature_names
    assert len(model.detector.rpn.anchor_generator.sizes) == 4
    assert len(model.detector.rpn.anchor_generator.aspect_ratios) == 4

    # Verify the multi-box target remains multi-box after the standard transform.
    _, transformed_targets = model.detector.transform(images, targets)
    assert transformed_targets is not None
    assert transformed_targets[0]["boxes"].shape == (0, 4)
    assert transformed_targets[1]["boxes"].shape == (1, 4)
    assert transformed_targets[2]["boxes"].shape == (2, 4)
    assert transformed_targets[3]["boxes"].shape == (1, 4)

    backbone_calls = 0
    rpn_feature_names: list[tuple[str, ...]] = []
    detection_roi_feature_names: list[tuple[str, ...]] = []
    seed_roi_feature_names: list[tuple[str, ...]] = []

    def count_backbone_calls(_module, _inputs, _output) -> None:
        nonlocal backbone_calls
        backbone_calls += 1

    def capture_rpn_features(_module, inputs) -> None:
        rpn_feature_names.append(tuple(inputs[1]))

    def capture_detection_roi_features(_module, inputs) -> None:
        detection_roi_feature_names.append(tuple(inputs[0]))

    def capture_seed_roi_features(_module, inputs) -> None:
        seed_roi_feature_names.append(tuple(inputs[0]))

    handle = model.backbone.register_forward_hook(count_backbone_calls)
    rpn_handle = model.detector.rpn.register_forward_pre_hook(capture_rpn_features)
    detection_roi_handle = model.detector.roi_heads.box_roi_pool.register_forward_pre_hook(
        capture_detection_roi_features
    )
    seed_roi_handle = model.seed_roi_pool.register_forward_pre_hook(capture_seed_roi_features)
    model.train()
    losses = model(
        images,
        targets=targets,
        seed_boxes=seed_boxes,
        attributes=attributes,
        seed_labels=seed_labels,
    )
    assert backbone_calls == 1
    expected_losses = {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_seed_cls",
    }
    assert set(losses) == expected_losses
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    total_loss = sum(losses.values())
    total_loss.backward()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())

    backbone_calls = 0
    model.eval()
    with torch.no_grad():
        output = model(images, seed_boxes=seed_boxes, attributes=attributes)
    handle.remove()
    rpn_handle.remove()
    detection_roi_handle.remove()
    seed_roi_handle.remove()
    assert backbone_calls == 1
    assert rpn_feature_names == [expected_feature_names, expected_feature_names]
    assert detection_roi_feature_names == [expected_feature_names, expected_feature_names]
    assert seed_roi_feature_names == [expected_feature_names, expected_feature_names]
    assert set(output) == {"detections", "seed_logits", "seed_probabilities"}
    assert len(output["detections"]) == 4
    assert output["seed_logits"].shape == (4,)
    assert output["seed_probabilities"].shape == (4,)
    assert torch.all((output["seed_probabilities"] >= 0) & (output["seed_probabilities"] <= 1))
    for detection in output["detections"]:
        assert set(detection) == {"boxes", "labels", "scores"}
        assert detection["boxes"].ndim == 2 and detection["boxes"].shape[1] == 4
        assert detection["scores"].shape == detection["labels"].shape
    # With a zero score threshold the random detector still emits proposals;
    # seed label semantics remain independent of the far-away detection target.
    assert output["detections"][3]["boxes"].shape[0] > 0
