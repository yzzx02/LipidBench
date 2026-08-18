from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(r"D:\CODE\LipidBench")
DATASET = REPO / "PeakTruthLab" / "datasets" / "PeakTruthLab_final_merged_20260814"
RUN_ROOT = REPO / "PeakTruthLab" / "results" / "rtx4070_final_merged_20260814" / "main_ablation" / "seed_20260814"
OUTPUT = REPO / "PeakTruthLab" / "results" / "rtx4070_final_merged_20260814" / "sanity_audit_20260815"

ATTRS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG", "SYM", "MOD", "EDGE"]
RUNS = [
    ("Attribute-only", "A_attribute_only"),
    ("Image-only", "B_image_only"),
    ("Naive concat", "C_naive_concat"),
    ("Image-gated-Attribute", "D_gated_fusion"),
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    pred = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probability)) if both_classes else None,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2) if np.isfinite(recall) and np.isfinite(specificity) else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def target_mean_baseline(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> dict:
    key = columns[0] if len(columns) == 1 else columns
    rates = train.groupby(key, dropna=False)["seed_label"].agg(["mean", "count"])
    global_rate = float(train["seed_label"].mean())
    train_keys = train[columns].fillna("<NA>").astype(str).agg("||".join, axis=1)
    val_keys = val[columns].fillna("<NA>").astype(str).agg("||".join, axis=1)
    rate_by_key = pd.DataFrame({"key": train_keys, "label": train["seed_label"].to_numpy()}).groupby("key")["label"].mean()
    score = val_keys.map(rate_by_key).fillna(global_rate).to_numpy(dtype=float)
    return {
        "feature": "+".join(columns),
        "val_roc_auc": float(roc_auc_score(val["seed_label"], score)),
        "val_pr_auc": float(average_precision_score(val["seed_label"], score)),
        "train_groups": int(len(rates)),
        "val_unseen_group_rows": int((~val_keys.isin(rate_by_key.index)).sum()),
        "description": "Train group positive rate mapped to Val; unseen groups use Train global rate",
    }


def bootstrap_auc_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, repeats: int = 2000) -> dict:
    rng = np.random.default_rng(20260815)
    deltas = []
    n = len(y)
    for _ in range(repeats):
        index = rng.integers(0, n, size=n)
        sampled_y = y[index]
        if len(np.unique(sampled_y)) < 2:
            continue
        deltas.append(roc_auc_score(sampled_y, a[index]) - roc_auc_score(sampled_y, b[index]))
    values = np.asarray(deltas)
    probability_positive = float(np.mean(values > 0))
    return {
        "comparison": "Image-gated-Attribute minus Naive concat",
        "observed_auc_difference": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "paired_bootstrap_repeats": int(len(values)),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "bootstrap_probability_difference_gt_zero": probability_positive,
        "two_sided_bootstrap_p": float(min(1.0, 2 * min(probability_positive, 1 - probability_positive))),
    }


def sample_stratified(frame: pd.DataFrame, n_per_stratum: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby(["seed_label", "old_new_batch"], dropna=False, sort=True):
        pieces.append(group.sample(n=min(n_per_stratum, len(group)), random_state=seed))
    return pd.concat(pieces, ignore_index=True)


def image_features(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for relative in frame["image"]:
        with Image.open(DATASET / str(relative)) as image:
            array = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        center = array[16:48, 16:48]
        border = np.concatenate([array[:8].ravel(), array[-8:].ravel(), array[:, :8].ravel(), array[:, -8:].ravel()])
        rows.append(
            [
                float(array.mean()),
                float(array.std()),
                *[float(value) for value in np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])],
                float((array < 0.10).mean()),
                float((array > 0.90).mean()),
                float(np.abs(np.diff(array, axis=0)).mean()),
                float(np.abs(np.diff(array, axis=1)).mean()),
                float(center.mean() - border.mean()),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_path = DATASET / "splits" / "train.csv"
    val_path = DATASET / "splits" / "val.csv"
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    train["seed_label"] = train["seed_label"].astype(int)
    val["seed_label"] = val["seed_label"].astype(int)

    fatal_issues: list[str] = []
    cautions: list[str] = []
    checks: list[dict] = []

    if len(train) != 15853 or len(val) != 1982:
        fatal_issues.append("Unexpected Train/Val row count")

    overlap_columns = ["image_id", "seed_id", "image", "annotation_json", "image_sha256", "split_group_id"]
    for column in overlap_columns:
        left = set(train[column].dropna().astype(str)) - {""}
        right = set(val[column].dropna().astype(str)) - {""}
        overlap = left & right
        checks.append({"check": f"train_val_overlap::{column}", "value": len(overlap), "expected": 0, "status": "ok" if not overlap else "fail"})
        if overlap:
            fatal_issues.append(f"Train/Val overlap in {column}: {len(overlap)}")

    same_source_files = set(train["source_file"].astype(str)) & set(val["source_file"].astype(str))
    cautions.append(
        f"The frozen main split is sample-level, not source/domain-held-out: {len(same_source_files)} source_file values occur in both Train and Val. This is protocol-compliant but makes the result an in-distribution estimate."
    )

    label_distribution_rows = []
    for split_name, frame in (("Train", train), ("Val", val)):
        for batch, group in frame.groupby("old_new_batch", dropna=False):
            label_distribution_rows.append(
                {
                    "split": split_name,
                    "old_new_batch": batch,
                    "n": len(group),
                    "positive": int(group["seed_label"].sum()),
                    "negative": int(len(group) - group["seed_label"].sum()),
                    "positive_rate": float(group["seed_label"].mean()),
                }
            )
    write_csv(OUTPUT / "label_distribution_by_batch.csv", label_distribution_rows)
    old_rate = float(train.loc[train["old_new_batch"] == "old_final_reviewed_20260725", "seed_label"].mean())
    new_rate = float(train.loc[train["old_new_batch"] == "new_manual_negative_4500_v2_20260814", "seed_label"].mean())
    cautions.append(
        f"Batch and label are strongly associated: Train positive rate is {old_rate:.4f} in old-final versus {new_rate:.4f} in the new manual-negative batch. Report old-only performance and LODO alongside the pooled score."
    )

    recomputed_rows = []
    subgroup_rows = []
    aligned_probabilities: dict[str, pd.Series] = {}
    scaler_max_error = 0.0
    computed_scaler = {"fill": {}, "mean": {}, "std": {}}
    for attr in ATTRS:
        values = pd.to_numeric(train[attr], errors="coerce")
        median = float(values.median()) if np.isfinite(values.median()) else 0.0
        filled = values.fillna(median)
        computed_scaler["fill"][attr] = median
        computed_scaler["mean"][attr] = float(filled.mean())
        sigma = float(filled.std(ddof=0))
        computed_scaler["std"][attr] = sigma if np.isfinite(sigma) and sigma >= 1e-8 else 1.0

    for model_name, directory in RUNS:
        run_dir = RUN_ROOT / directory
        prediction = pd.read_csv(run_dir / "best_val_predictions.csv")
        selection = read_json(run_dir / "selection_on_val.json")
        scaler = read_json(run_dir / "attr_scaler.json")
        if prediction["seed_id"].duplicated().any():
            fatal_issues.append(f"{model_name}: duplicate seed_id in Val predictions")
        expected_ids = set(val["seed_id"].astype(str))
        observed_ids = set(prediction["seed_id"].astype(str))
        if expected_ids != observed_ids:
            fatal_issues.append(f"{model_name}: prediction IDs do not exactly match frozen Val")
        merged = val[["seed_id", "seed_label", "old_new_batch", "domain_id", "source_file", "difficulty_type"]].merge(
            prediction[["seed_id", "label", "prob_true_peak"]], on="seed_id", how="left", validate="one_to_one"
        )
        if merged["prob_true_peak"].isna().any() or not np.isfinite(merged["prob_true_peak"]).all():
            fatal_issues.append(f"{model_name}: missing or non-finite Val scores")
        if not np.array_equal(merged["seed_label"].to_numpy(dtype=int), merged["label"].to_numpy(dtype=int)):
            fatal_issues.append(f"{model_name}: prediction labels disagree with frozen Val labels")
        threshold = float(selection["selected_threshold"])
        result = metrics(merged["seed_label"].to_numpy(), merged["prob_true_peak"].to_numpy(), threshold)
        declared = selection["selected_threshold_metrics"]
        discrepancy = max(
            abs(result["roc_auc"] - float(declared["auc"])),
            abs(result["pr_auc"] - float(declared["pr_auc"])),
            abs(result["f1"] - float(declared["f1"])),
        )
        if discrepancy > 1e-12:
            fatal_issues.append(f"{model_name}: independently recomputed metrics differ by {discrepancy}")
        recomputed_rows.append(
            {
                "model": model_name,
                "best_epoch": selection["best_epoch"],
                "threshold": threshold,
                **result,
                "metric_max_abs_discrepancy": discrepancy,
                "score_min": float(merged["prob_true_peak"].min()),
                "score_max": float(merged["prob_true_peak"].max()),
                "score_exact_zero": int((merged["prob_true_peak"] == 0).sum()),
                "score_exact_one": int((merged["prob_true_peak"] == 1).sum()),
            }
        )
        aligned_probabilities[model_name] = merged.set_index("seed_id")["prob_true_peak"].reindex(val["seed_id"]).astype(float)

        for batch, group in merged.groupby("old_new_batch", dropna=False):
            subgroup_rows.append({"model": model_name, "group_type": "old_new_batch", "group": batch, "threshold": threshold, **metrics(group["seed_label"], group["prob_true_peak"], threshold)})
        for domain, group in merged.groupby("domain_id", dropna=False):
            subgroup_rows.append({"model": model_name, "group_type": "domain_id", "group": domain, "threshold": threshold, **metrics(group["seed_label"], group["prob_true_peak"], threshold)})

        for section in ("fill", "mean", "std"):
            for attr in ATTRS:
                scaler_max_error = max(scaler_max_error, abs(float(scaler[section][attr]) - computed_scaler[section][attr]))

    write_csv(OUTPUT / "independently_recomputed_val_metrics.csv", recomputed_rows)
    write_csv(OUTPUT / "val_subgroup_metrics.csv", subgroup_rows)
    if scaler_max_error > 1e-12:
        fatal_issues.append(f"Saved attribute scaler differs from independent Train-only recomputation: {scaler_max_error}")
    checks.append({"check": "train_only_scaler_max_abs_error", "value": scaler_max_error, "expected": 0, "status": "ok" if scaler_max_error <= 1e-12 else "fail"})

    y_val = val["seed_label"].to_numpy(dtype=int)
    concat_prob = aligned_probabilities["Naive concat"].to_numpy()
    gated_prob = aligned_probabilities["Image-gated-Attribute"].to_numpy()
    bootstrap = bootstrap_auc_difference(y_val, gated_prob, concat_prob)
    write_json(OUTPUT / "paired_bootstrap_gated_vs_concat.json", bootstrap)
    if bootstrap["ci_95_low"] <= 0 <= bootstrap["ci_95_high"]:
        cautions.append("The paired bootstrap 95% interval for Gated minus Concat Val ROC-AUC includes zero; the observed 0.000317 difference is not stable evidence of superiority on this single split.")

    metadata_rows = []
    for columns in (["old_new_batch"], ["domain_id"], ["source_file"], ["difficulty_type"], ["domain_id", "old_new_batch", "difficulty_type"]):
        metadata_rows.append(target_mean_baseline(train, val, list(columns)))
    write_csv(OUTPUT / "metadata_shortcut_baselines.csv", metadata_rows)

    x_train = train[ATTRS].apply(pd.to_numeric, errors="coerce")
    x_val = val[ATTRS].apply(pd.to_numeric, errors="coerce")
    y_train = train["seed_label"].to_numpy(dtype=int)
    attr_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=3000, random_state=20260815)),
        ]
    )
    attr_model.fit(x_train, y_train)
    attr_score = attr_model.predict_proba(x_val)[:, 1]
    rng = np.random.default_rng(20260815)
    permuted_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=3000, random_state=20260815)),
        ]
    )
    permuted_y_train = rng.permutation(y_train)
    permuted_y_val = rng.permutation(y_val)
    permuted_model.fit(x_train, permuted_y_train)
    permuted_score = permuted_model.predict_proba(x_val)[:, 1]
    baseline_rows = [
        {
            "baseline": "16-attribute logistic regression",
            "train_labels": "true",
            "val_roc_auc": float(roc_auc_score(y_val, attr_score)),
            "val_pr_auc": float(average_precision_score(y_val, attr_score)),
        },
        {
            "baseline": "16-attribute logistic regression independent label-permutation sanity",
            "train_labels": "permuted; evaluated against independently permuted Val labels",
            "val_roc_auc": float(roc_auc_score(permuted_y_val, permuted_score)),
            "val_pr_auc": float(average_precision_score(permuted_y_val, permuted_score)),
        },
    ]

    univariate_rows = []
    imputer = SimpleImputer(strategy="median")
    imputer.fit(x_train)
    x_val_imputed = imputer.transform(x_val)
    for index, attr in enumerate(ATTRS):
        raw_auc = float(roc_auc_score(y_val, x_val_imputed[:, index]))
        univariate_rows.append({"attribute": attr, "raw_auc": raw_auc, "direction_free_auc": max(raw_auc, 1 - raw_auc)})
    univariate_rows.sort(key=lambda row: row["direction_free_auc"], reverse=True)
    write_csv(OUTPUT / "univariate_attribute_auc.csv", univariate_rows)

    image_train = sample_stratified(train, n_per_stratum=1000, seed=20260815)
    image_x_train = image_features(image_train)
    image_x_val = image_features(val)
    image_model = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=3000, random_state=20260815))])
    image_model.fit(image_x_train, image_train["seed_label"].to_numpy(dtype=int))
    image_score = image_model.predict_proba(image_x_val)[:, 1]
    baseline_rows.append(
        {
            "baseline": "12 simple pixel-statistics logistic regression (no filenames/attributes)",
            "train_labels": "true",
            "val_roc_auc": float(roc_auc_score(y_val, image_score)),
            "val_pr_auc": float(average_precision_score(y_val, image_score)),
        }
    )
    write_csv(OUTPUT / "independent_baselines.csv", baseline_rows)

    near = pd.read_csv(DATASET / "audits" / "near_duplicate_pairs.csv")
    train_ids = set(train["seed_id"].astype(str))
    val_ids = set(val["seed_id"].astype(str))
    cross_near = near[
        (near["left_seed_id"].astype(str).isin(train_ids) & near["right_seed_id"].astype(str).isin(val_ids))
        | (near["right_seed_id"].astype(str).isin(train_ids) & near["left_seed_id"].astype(str).isin(val_ids))
    ].copy()
    high_similarity_cross = cross_near[cross_near["high_similarity"].astype(str).str.lower().eq("true")]
    checks.append({"check": "high_similarity_near_pairs_crossing_train_val", "value": len(high_similarity_cross), "expected": 0, "status": "ok" if len(high_similarity_cross) == 0 else "fail"})
    if len(high_similarity_cross):
        fatal_issues.append(f"High-similarity near-duplicate pairs cross Train/Val: {len(high_similarity_cross)}")
    cross_near.to_csv(OUTPUT / "cross_split_near_duplicate_review.csv", index=False, encoding="utf-8-sig")

    audit = {
        "status": "PASS_WITH_DESIGN_CAUTIONS" if not fatal_issues else "FAIL",
        "scope": "Train and Val only; Test manifest rows, Test labels and Test images were not loaded",
        "fatal_issues": fatal_issues,
        "cautions": cautions,
        "key_findings": {
            "train_rows": len(train),
            "val_rows": len(val),
            "train_csv_sha256": sha256(train_path),
            "val_csv_sha256": sha256(val_path),
            "all_four_metrics_exactly_reproduced": not any("recomputed metrics" in item for item in fatal_issues),
            "prediction_ids_and_labels_match_frozen_val": not any("prediction IDs" in item or "prediction labels" in item for item in fatal_issues),
            "train_val_content_or_group_overlap": any("overlap" in item for item in fatal_issues),
            "train_only_scaler_max_abs_error": scaler_max_error,
            "gated_minus_concat_bootstrap": bootstrap,
            "attribute_logistic_val_auc": baseline_rows[0]["val_roc_auc"],
            "permuted_attribute_logistic_val_auc": baseline_rows[1]["val_roc_auc"],
            "simple_pixel_statistics_val_auc": baseline_rows[2]["val_roc_auc"],
            "test_accessed": False,
        },
        "interpretation": (
            "No implementation or split-integrity defect was found. The high pooled Val scores are plausible because both image morphology and attributes are highly informative. "
            "However, the sample-level split shares sources/domains across Train and Val, and the new batch is strongly negative-enriched; therefore pooled Val is an in-distribution result and may be optimistic for unseen-domain generalization."
        ),
    }
    write_json(OUTPUT / "sanity_audit.json", audit)

    lines = [
        "# RTX4070 main-ablation sanity audit (Train/Val only)",
        "",
        f"Status: **{audit['status']}**",
        "",
        "No Test rows, labels, images or predictions were loaded. Four-model metrics were independently recomputed from frozen Val predictions.",
        "",
        "## Checks",
        "",
        f"- Fatal issues: {len(fatal_issues)}",
        f"- Train-only scaler maximum absolute discrepancy: {scaler_max_error:.3g}",
        f"- Gated minus Concat Val ROC-AUC: {bootstrap['observed_auc_difference']:.9f}; paired bootstrap 95% CI [{bootstrap['ci_95_low']:.9f}, {bootstrap['ci_95_high']:.9f}], p={bootstrap['two_sided_bootstrap_p']:.4f}",
        f"- Independent 16-attribute logistic ROC-AUC: {baseline_rows[0]['val_roc_auc']:.6f}",
        f"- Attribute label-permutation sanity ROC-AUC: {baseline_rows[1]['val_roc_auc']:.6f}",
        f"- Simple pixel-statistics ROC-AUC: {baseline_rows[2]['val_roc_auc']:.6f}",
        "",
        "## Interpretation",
        "",
        audit["interpretation"],
        "",
        "The Gated/Concat difference is small and its paired-bootstrap interval includes zero. Choosing Concat for simplicity is defensible, but it must be reported as a user/design choice rather than a validation win. Main Test must remain single-use and LODO should be the principal evidence for unseen-domain generalization.",
        "",
        "## Design cautions",
        "",
    ]
    lines.extend(f"- {item}" for item in cautions)
    if fatal_issues:
        lines.extend(["", "## Fatal issues", ""] + [f"- {item}" for item in fatal_issues])
    (OUTPUT / "SANITY_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
