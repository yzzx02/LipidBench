from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


ATTRS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG", "SYM", "MOD", "EDGE"]
SEED = 20260814
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

SOURCE_DOMAINS = {
    "0001_MAY_RoCI-StEM_CP-287": ("ST002903", "Identification and targeting of microbial putrescine acetylation in bloodstream infections"),
    "0021_MAY_ROCI-StEM_HN_575": ("ST002903", "Identification and targeting of microbial putrescine acetylation in bloodstream infections"),
    "060-0145-005_017": ("ST004085", "Methionine restriction alters lipid beta-oxidation and levels of lipids associated with CVD risk"),
    "060-0145-006_018": ("ST004085", "Methionine restriction alters lipid beta-oxidation and levels of lipids associated with CVD risk"),
    "20180321_S00033936_P": ("ST002135", "Alignment and Analysis of a Disparately Acquired Multi-Batch Metabolomics Study of Maternal Pregnancy Samples (Part 2)"),
    "20180323_S00033882_N": ("ST002135", "Alignment and Analysis of a Disparately Acquired Multi-Batch Metabolomics Study of Maternal Pregnancy Samples (Part 2)"),
    "6545_20201012_14_+_QC 01": ("ST002162", "CFAP418 participates in membrane-associated cellular processes through binding lipids during ciliogenesis"),
    "6545_20201012_22_+_QC 03": ("ST002162", "CFAP418 participates in membrane-associated cellular processes through binding lipids during ciliogenesis"),
    "6545A_20230727_-_57_18": ("ST003220", "Obesity, sex, and depot drive distinct lipid profiles in murine white adipose tissue"),
    "6545A_20230729_+_07_9": ("ST003220", "Obesity, sex, and depot drive distinct lipid profiles in murine white adipose tissue"),
    "AG-88-11_r12-": ("ST003792", "Untargeted lipidomics characterization of blood samples from patients with maple syrup urine disease"),
    "AG-88-11_r2+": ("ST003792", "Untargeted lipidomics characterization of blood samples from patients with maple syrup urine disease"),
    "Blood-15V": ("ST001873", "Metabolomics analysis of multiple samples on AB 5600-Part 1"),
    "Blood-30V": ("ST001873", "Metabolomics analysis of multiple samples on AB 5600-Part 1"),
    "HepG2-30V": ("ST001873", "Metabolomics analysis of multiple samples on AB 5600-Part 1"),
    "Urine-15V": ("ST001873", "Metabolomics analysis of multiple samples on AB 5600-Part 1"),
    "Urine-30V": ("ST001873", "Metabolomics analysis of multiple samples on AB 5600-Part 1"),
    "D03P_POS": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "D04P_NEG": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "D06P_POS": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "D07P_NEG": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "NIST_Full scan_1_POS": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "NIST_Full scan_2_NEG": ("ST003514", "Highly reliable LC-MS lipidomics database for efficient human plasma profiling based on NIST SRM 1950"),
    "HetCD_mixneg": ("ST003127", "Effect of High Fat Diet on Heart Lipidome in CHCHD10 Mutant Mice"),
    "WTHFD_mixpos": ("ST003127", "Effect of High Fat Diet on Heart Lipidome in CHCHD10 Mutant Mice"),
    "Sphingolipid-BTSPT-Cell-pellet_1": ("ST003941", "Lipidomic analysis of Bacteroides thetaiotaomicron WT vs SPT KO"),
    "Sphingolipid-BTSPT-OMV_1": ("ST003941", "Lipidomic analysis of Bacteroides thetaiotaomicron WT vs SPT KO"),
    "Sphingolipid-BTWT-Cell-pellet_1": ("ST003941", "Lipidomic analysis of Bacteroides thetaiotaomicron WT vs SPT KO"),
    "frag1_pos20_1": ("SELF_FRAG1_20260527", "frag1 self-collected dataset"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def finite_or_nan(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) and np.isfinite(number) else float("nan")


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.uint8)
    comparisons = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit in comparisons.ravel():
        value = (value << 1) | int(bit)
    return value


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def create(cls, values: Iterable[str]) -> "UnionFind":
        return cls(parent={value: value for value in values})

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def load_old(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    old_root = Path(args.old_root)
    manifests: list[dict] = []
    for old_split in ("train", "val", "test"):
        for row in read_jsonl(old_root / "manifests" / f"{old_split}.jsonl"):
            row["old_split"] = old_split
            manifests.append(row)
    manifest_by_id = {str(row["sample_id"]): row for row in manifests}
    if len(manifest_by_id) != 15317:
        raise RuntimeError(f"Expected 15317 unique old-final samples, got {len(manifest_by_id)}")

    seed_jobs = pd.read_csv(args.old_seed_jobs, dtype=str, keep_default_na=False)
    seed_calc = pd.read_csv(args.old_seed_calc, dtype=str, keep_default_na=False)
    peaks_jobs = pd.read_csv(args.old_peak_jobs, dtype=str, keep_default_na=False)
    peaks_calc = pd.read_csv(args.old_peak_calc, dtype=str, keep_default_na=False)
    seed = seed_jobs.merge(seed_calc, on=["job_id", "entity_type", "sample_id", "source_mzml_name"], validate="one_to_one", suffixes=("_job", ""))
    peaks = peaks_jobs.merge(peaks_calc, on=["job_id", "entity_type", "sample_id", "source_mzml_name"], validate="one_to_one", suffixes=("_job", ""))
    if len(seed) != 15317 or len(peaks) != 18580:
        raise RuntimeError(f"Old 16-attribute row mismatch: seed={len(seed)}, peaks={len(peaks)}")
    if (seed["error"].str.strip() != "").any() or (peaks["error"].str.strip() != "").any():
        raise RuntimeError("Old 16-attribute result contains job errors")

    unknown_sources = sorted(set(seed["source_file"]) - set(SOURCE_DOMAINS))
    if unknown_sources:
        raise RuntimeError(f"Missing independent-domain definitions: {unknown_sources}")

    image_paths: dict[str, Path] = {}
    records: list[dict] = []
    peak_grouped = {sample_id: group for sample_id, group in peaks.groupby("sample_id", sort=False)}
    for _, row in seed.iterrows():
        sample_id = str(row["sample_id"])
        manifest = manifest_by_id[sample_id]
        source = str(row["source_file"])
        domain_id, domain_name = SOURCE_DOMAINS[source]
        image_src = old_root / "eic_images_flat" / str(manifest["image_path"])
        if not image_src.is_file():
            raise FileNotFoundError(image_src)
        pgroup = peak_grouped.get(sample_id)
        true_count = int((pgroup["final_label"] == "True_Peak").sum()) if pgroup is not None else 0
        out_count = int((pgroup["final_label"] == "OUT_FIG").sum()) if pgroup is not None else 0
        image_rel = Path("images") / "old" / source / f"{sample_id}.png"
        json_rel = image_rel.with_suffix(".json")
        seed_id = f"OLD::{sample_id}"
        image_paths[seed_id] = image_src
        record = {
            "image_id": "",
            "seed_id": seed_id,
            "original_image_id": row["image_id"],
            "original_seed_id": sample_id,
            "old_new_batch": "old_final_reviewed_20260725",
            "batch_name": "paper_final_reviewed_20260725",
            "old_split_provenance_only": row["old_split"],
            "source_file": source,
            "source_mzml_name": row["source_mzml_name"],
            "source_mzml_sha256": row["source_mzml_sha256"],
            "domain_id": domain_id,
            "domain_name": domain_name,
            "mz": finite_or_nan(row["mz"]),
            "seed_rt_min": finite_or_nan(row["RT"]),
            "seed_left_rt_min": finite_or_nan(row["RTmin"]),
            "seed_right_rt_min": finite_or_nan(row["RTmax"]),
            "eic_tolerance_ppm": finite_or_nan(row["eic_tolerance_ppm"]),
            "seed_label": int(row["final_seed_label"]),
            "has_human_box": bool(true_count + out_count),
            "true_peak_count": true_count,
            "out_fig_count": out_count,
            "human_box_count": true_count + out_count,
            "seed_relation": manifest.get("metadata", {}).get("seed_relation", ""),
            "difficulty_type": "not_applicable_old_final",
            "reviewed_sample": bool_value(row["reviewed_sample"]),
            "image": image_rel.as_posix(),
            "annotation_json": json_rel.as_posix(),
            "seed_box_json": canonical_json(manifest.get("seed_box", [])),
            "attribute_provenance": "old_final_13_preserved_plus_remote_SYM_MOD_EDGE",
        }
        for attr in ATTRS:
            record[attr] = finite_or_nan(row[attr])
        records.append(record)

    peak_records: list[dict] = []
    for _, row in peaks.iterrows():
        sample_id = str(row["sample_id"])
        source = str(row["source_file"])
        domain_id, domain_name = SOURCE_DOMAINS[source]
        peak_id = f"OLD::{row['job_id']}"
        item = {
            "peak_instance_id": peak_id,
            "original_peak_instance_id": row["job_id"],
            "seed_id": f"OLD::{sample_id}",
            "original_seed_id": sample_id,
            "image_id": "",
            "old_new_batch": "old_final_reviewed_20260725",
            "batch_name": "paper_final_reviewed_20260725",
            "source_file": source,
            "source_mzml_name": row["source_mzml_name"],
            "source_mzml_sha256": row["source_mzml_sha256"],
            "domain_id": domain_id,
            "domain_name": domain_name,
            "label": row["final_label"],
            "is_out_fig": row["final_label"] == "OUT_FIG",
            "box_index": int(row["final_box_index"]),
            "box_xmin_px": finite_or_nan(row["final_box_xmin_px"]),
            "box_ymin_px": finite_or_nan(row["final_box_ymin_px"]),
            "box_xmax_px": finite_or_nan(row["final_box_xmax_px"]),
            "box_ymax_px": finite_or_nan(row["final_box_ymax_px"]),
            "peak_left_rt_min": finite_or_nan(row["RTmin"]),
            "peak_apex_rt_min": finite_or_nan(row["RT"]),
            "peak_right_rt_min": finite_or_nan(row["RTmax"]),
            "reviewed_sample": bool_value(row["reviewed_sample"]),
            "included_in_old_v2_manifest": bool_value(row["included_in_v2_manifest"]),
            "box_source": row["box_source"],
            "image": (Path("images") / "old" / source / f"{sample_id}.png").as_posix(),
            "annotation_json": (Path("images") / "old" / source / f"{sample_id}.json").as_posix(),
            "attribute_provenance": row["attribute_policy"],
        }
        for attr in ATTRS:
            item[attr] = finite_or_nan(row[attr])
        peak_records.append(item)
    return pd.DataFrame(records), pd.DataFrame(peak_records), image_paths


def load_new(args: argparse.Namespace, source_hashes: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path], dict[str, Path]]:
    new_root = Path(args.new_root)
    seed = pd.read_csv(new_root / "seed_attributes.csv", dtype=str, keep_default_na=False)
    peaks = pd.read_csv(new_root / "true_peak_instances.csv", dtype=str, keep_default_na=False)
    if len(seed) != 4500 or len(peaks) != 335:
        raise RuntimeError(f"New package row mismatch: seed={len(seed)}, peaks={len(peaks)}")

    png_by_stem: dict[str, Path] = {}
    json_by_stem: dict[str, Path] = {}
    for path in (new_root / "corrected_annotations").rglob("*.png"):
        if path.stem in png_by_stem:
            raise RuntimeError(f"Duplicate new PNG stem: {path.stem}")
        png_by_stem[path.stem] = path
    for path in (new_root / "corrected_annotations").rglob("*.json"):
        if path.stem in json_by_stem:
            raise RuntimeError(f"Duplicate new JSON stem: {path.stem}")
        json_by_stem[path.stem] = path
    if len(png_by_stem) != 4500 or len(json_by_stem) != 4500:
        raise RuntimeError(f"New PNG/JSON mismatch: {len(png_by_stem)}/{len(json_by_stem)}")

    peak_counts = peaks.groupby("原始特征编号")["原始标签"].agg(list).to_dict()
    image_paths: dict[str, Path] = {}
    json_paths: dict[str, Path] = {}
    records: list[dict] = []
    for _, row in seed.iterrows():
        sample_id = str(row["原始特征编号"])
        source = Path(str(row["mzML文件"])).stem
        if source not in SOURCE_DOMAINS:
            raise RuntimeError(f"Missing independent-domain definition for new source: {source}")
        domain_id, domain_name = SOURCE_DOMAINS[source]
        batch = json_by_stem[sample_id].parents[1].name
        labels = peak_counts.get(sample_id, [])
        true_count = sum(label == "True_Peak" for label in labels)
        out_count = sum(label == "OUT_FIG" for label in labels)
        image_rel = Path("images") / "new" / batch / f"{sample_id}.png"
        json_rel = image_rel.with_suffix(".json")
        seed_id = f"NEW::{sample_id}"
        image_paths[seed_id] = png_by_stem[sample_id]
        json_paths[seed_id] = json_by_stem[sample_id]
        difficulty = sample_id.split("__")[1] if len(sample_id.split("__")) >= 3 else "UNPARSED"
        record = {
            "image_id": "",
            "seed_id": seed_id,
            "original_image_id": row["图片编号"],
            "original_seed_id": sample_id,
            "old_new_batch": "new_manual_negative_4500_v2_20260814",
            "batch_name": batch,
            "old_split_provenance_only": "",
            "source_file": source,
            "source_mzml_name": f"{source}.mzML",
            "source_mzml_sha256": source_hashes.get(source, ""),
            "domain_id": domain_id,
            "domain_name": domain_name,
            "mz": finite_or_nan(row["m/z"]),
            "seed_rt_min": finite_or_nan(row["原始种子RT"]),
            "seed_left_rt_min": finite_or_nan(row["原始左边界RT"]),
            "seed_right_rt_min": finite_or_nan(row["原始右边界RT"]),
            "eic_tolerance_ppm": finite_or_nan(row["EIC提取容差_ppm"]),
            "seed_label": int(row["种子真假标签"]),
            "has_human_box": bool(true_count + out_count),
            "true_peak_count": true_count,
            "out_fig_count": out_count,
            "human_box_count": true_count + out_count,
            "seed_relation": row["Seed关系类型"],
            "difficulty_type": difficulty,
            "reviewed_sample": True,
            "image": image_rel.as_posix(),
            "annotation_json": json_rel.as_posix(),
            "seed_box_json": row["原始Seed框坐标_px"],
            "attribute_provenance": "manual_negative_4500_v2_original_16",
        }
        for attr in ATTRS:
            record[attr] = finite_or_nan(row[f"原始{attr}"])
        records.append(record)

    peak_records: list[dict] = []
    for _, row in peaks.iterrows():
        sample_id = str(row["原始特征编号"])
        source = Path(str(row["mzML文件"])).stem
        domain_id, domain_name = SOURCE_DOMAINS[source]
        batch = json_by_stem[sample_id].parents[1].name
        item = {
            "peak_instance_id": f"NEW::{row['真峰编号']}",
            "original_peak_instance_id": row["真峰编号"],
            "seed_id": f"NEW::{sample_id}",
            "original_seed_id": sample_id,
            "image_id": "",
            "old_new_batch": "new_manual_negative_4500_v2_20260814",
            "batch_name": batch,
            "source_file": source,
            "source_mzml_name": f"{source}.mzML",
            "source_mzml_sha256": source_hashes.get(source, ""),
            "domain_id": domain_id,
            "domain_name": domain_name,
            "label": row["原始标签"],
            "is_out_fig": bool_value(row["是否OUT_FIG"]),
            "box_index": int(str(row["真峰编号"]).rsplit("P", 1)[-1]),
            "box_xmin_px": finite_or_nan(row["修正后框_xmin_px"]),
            "box_ymin_px": finite_or_nan(row["修正后框_ymin_px"]),
            "box_xmax_px": finite_or_nan(row["修正后框_xmax_px"]),
            "box_ymax_px": finite_or_nan(row["修正后框_ymax_px"]),
            "peak_left_rt_min": finite_or_nan(row["修正后左边界RT"]),
            "peak_apex_rt_min": finite_or_nan(row["峰顶RT"]),
            "peak_right_rt_min": finite_or_nan(row["修正后右边界RT"]),
            "reviewed_sample": True,
            "included_in_old_v2_manifest": False,
            "box_source": "manual_negative_4500_v2_corrected_labelme",
            "image": (Path("images") / "new" / batch / f"{sample_id}.png").as_posix(),
            "annotation_json": (Path("images") / "new" / batch / f"{sample_id}.json").as_posix(),
            "attribute_provenance": "manual_negative_4500_v2_original_16",
        }
        for attr in ATTRS:
            item[attr] = finite_or_nan(row[attr])
        peak_records.append(item)
    return pd.DataFrame(records), pd.DataFrame(peak_records), image_paths, json_paths


def assign_ids_and_copy(seed_df: pd.DataFrame, peak_df: pd.DataFrame, image_paths: dict[str, Path], new_json_paths: dict[str, Path], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    seed_df = seed_df.sort_values(["old_new_batch", "source_file", "original_seed_id"], kind="stable").reset_index(drop=True)
    seed_df["image_id"] = [f"PTL-{index:06d}" for index in range(1, len(seed_df) + 1)]
    image_id_by_seed = dict(zip(seed_df["seed_id"], seed_df["image_id"], strict=True))
    peak_df["image_id"] = peak_df["seed_id"].map(image_id_by_seed)

    peak_by_seed = {seed_id: group.sort_values("box_index") for seed_id, group in peak_df.groupby("seed_id", sort=False)}
    file_manifest: list[dict] = []
    image_hashes: list[str] = []
    annotation_hashes: list[str] = []
    dhashes: list[int] = []
    for _, row in seed_df.iterrows():
        seed_id = row["seed_id"]
        image_src = image_paths[seed_id]
        image_dst = output_root / row["image"]
        json_dst = output_root / row["annotation_json"]
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_dst)

        if seed_id.startswith("OLD::"):
            shapes = []
            for _, peak in peak_by_seed.get(seed_id, pd.DataFrame()).iterrows():
                shapes.append(
                    {
                        "label": str(peak["label"]),
                        "points": [[float(peak["box_xmin_px"]), float(peak["box_ymin_px"])], [float(peak["box_xmax_px"]), float(peak["box_ymax_px"])]],
                        "group_id": None,
                        "description": "authoritative old-final annotation",
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None,
                    }
                )
            payload = {
                "version": "7.0.4",
                "flags": {"provenance": "paper_final_reviewed_20260725"},
                "shapes": shapes,
                "imagePath": image_dst.name,
                "imageData": None,
                "imageHeight": 480,
                "imageWidth": 480,
            }
        else:
            payload = json.loads(new_json_paths[seed_id].read_text(encoding="utf-8"))
            payload["imagePath"] = image_dst.name
            payload["imageData"] = None
        write_json(json_dst, payload)

        image_sha = sha256_file(image_dst)
        annotation_sha = sha256_file(json_dst)
        image_hashes.append(image_sha)
        annotation_hashes.append(annotation_sha)
        dhashes.append(dhash64(image_dst))
        file_manifest.extend(
            [
                {"relative_path": row["image"], "file_type": "PNG", "sha256": image_sha, "size_bytes": image_dst.stat().st_size},
                {"relative_path": row["annotation_json"], "file_type": "LabelMe JSON", "sha256": annotation_sha, "size_bytes": json_dst.stat().st_size},
            ]
        )
    seed_df["image_sha256"] = image_hashes
    seed_df["annotation_sha256"] = annotation_hashes
    seed_df["image_dhash64"] = [f"{value:016x}" for value in dhashes]
    return seed_df, peak_df, file_manifest


def audit_duplicates(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    if seed_df["seed_id"].duplicated().any() or seed_df["original_seed_id"].duplicated().any():
        raise RuntimeError("Exact duplicate ID found; stop for manual resolution")
    if seed_df["image_sha256"].duplicated().any():
        raise RuntimeError("Exact duplicate image SHA256 found; stop for manual resolution")

    uf = UnionFind.create(seed_df["seed_id"])
    near_pairs: list[dict] = []
    by_source = {source: group.copy() for source, group in seed_df.groupby("source_file", sort=True)}
    for source, group in by_source.items():
        rows = group.sort_values(["mz", "seed_rt_min", "seed_id"]).to_dict("records")
        mz_values = [float(row["mz"]) for row in rows]
        for i, left in enumerate(rows):
            if not np.isfinite(left["mz"]) or not np.isfinite(left["seed_rt_min"]):
                continue
            upper = float(left["mz"]) * (1.0 + 5e-6)
            stop = bisect.bisect_right(mz_values, upper, lo=i + 1)
            for right in rows[i + 1 : stop]:
                if not np.isfinite(right["seed_rt_min"]):
                    continue
                ppm = abs(float(right["mz"]) - float(left["mz"])) / ((float(right["mz"]) + float(left["mz"])) / 2.0) * 1e6
                rt_seconds = abs(float(right["seed_rt_min"]) - float(left["seed_rt_min"])) * 60.0
                if ppm > 5.0 or rt_seconds > 2.0:
                    continue
                left_hash = int(str(left["image_dhash64"]), 16)
                right_hash = int(str(right["image_dhash64"]), 16)
                hamming = (left_hash ^ right_hash).bit_count()
                if hamming > 8:
                    continue
                high_similarity = hamming <= 4
                pair = {
                    "source_file": source,
                    "left_seed_id": left["seed_id"],
                    "left_batch": left["old_new_batch"],
                    "left_label": int(left["seed_label"]),
                    "right_seed_id": right["seed_id"],
                    "right_batch": right["old_new_batch"],
                    "right_label": int(right["seed_label"]),
                    "mz_ppm": ppm,
                    "rt_delta_seconds": rt_seconds,
                    "dhash_hamming": hamming,
                    "high_similarity": high_similarity,
                    "label_conflict": int(left["seed_label"]) != int(right["seed_label"]),
                }
                near_pairs.append(pair)
                if high_similarity:
                    uf.union(left["seed_id"], right["seed_id"])

    groups: dict[str, list[str]] = defaultdict(list)
    for seed_id in seed_df["seed_id"]:
        groups[uf.find(seed_id)].append(seed_id)
    non_singleton = sorted((sorted(members) for members in groups.values() if len(members) > 1), key=lambda members: members[0])
    duplicate_id_by_seed: dict[str, str] = {}
    for index, members in enumerate(non_singleton, start=1):
        group_id = f"DUP-{index:05d}"
        for seed_id in members:
            duplicate_id_by_seed[seed_id] = group_id
    seed_df["duplicate_group_id"] = seed_df["seed_id"].map(duplicate_id_by_seed).fillna("")
    seed_df["split_group_id"] = np.where(seed_df["duplicate_group_id"] != "", seed_df["duplicate_group_id"], seed_df["seed_id"])

    duplicate_rows: list[dict] = []
    conflict_rows: list[dict] = []
    for members in non_singleton:
        member_df = seed_df[seed_df["seed_id"].isin(members)]
        group_id = duplicate_id_by_seed[members[0]]
        conflict = member_df["seed_label"].nunique() > 1
        duplicate_rows.append(
            {
                "duplicate_group_id": group_id,
                "member_count": len(member_df),
                "seed_ids": "|".join(sorted(member_df["seed_id"])),
                "old_new_batches": "|".join(sorted(member_df["old_new_batch"].unique())),
                "source_files": "|".join(sorted(member_df["source_file"].unique())),
                "seed_labels": "|".join(str(x) for x in sorted(member_df["seed_label"].unique())),
                "label_conflict": conflict,
                "rule": "connected component of same-source <=5 ppm, <=2 s, dHash Hamming <=4",
            }
        )
        if conflict:
            conflict_rows.append(
                {
                    "conflict_id": f"CONFLICT-{len(conflict_rows)+1:04d}",
                    "duplicate_group_id": group_id,
                    "conflict_type": "near_duplicate_seed_label_disagreement",
                    "seed_ids": "|".join(sorted(member_df["seed_id"])),
                    "labels": "|".join(str(x) for x in sorted(member_df["seed_label"].unique())),
                    "resolution": "retain authoritative old-final labels; keep group intact; exclude entire group from Test",
                }
            )
    conflict_group_ids = {row["duplicate_group_id"] for row in conflict_rows}
    seed_df["conflict_status"] = np.where(seed_df["duplicate_group_id"].isin(conflict_group_ids), "near_duplicate_label_conflict", "none")
    seed_df["exclude_from_test"] = seed_df["duplicate_group_id"].isin(conflict_group_ids)
    exclusions = [
        {
            "seed_id": row.seed_id,
            "duplicate_group_id": row.duplicate_group_id,
            "reason": "near_duplicate_label_conflict_pending_manual_confirmation",
            "action": "excluded_from_test_only; retained in merged dataset with original authoritative label",
        }
        for row in seed_df[seed_df["exclude_from_test"]].itertuples()
    ]
    return seed_df, near_pairs, duplicate_rows, conflict_rows + exclusions


def target_sizes(total: int) -> dict[str, int]:
    floors = {split: int(math.floor(total * ratio)) for split, ratio in RATIOS.items()}
    remainder = total - sum(floors.values())
    fractions = sorted(RATIOS, key=lambda split: (-(total * RATIOS[split] - floors[split]), split))
    for split in fractions[:remainder]:
        floors[split] += 1
    return floors


def assign_split(seed_df: pd.DataFrame) -> pd.DataFrame:
    group_rows = []
    for group_id, group in seed_df.groupby("split_group_id", sort=True):
        group_rows.append(
            {
                "group_id": group_id,
                "size": len(group),
                "forbid_test": bool(group["exclude_from_test"].any()),
                "stratum": canonical_json(
                    {
                        "batch": sorted(group["old_new_batch"].unique()),
                        "source": sorted(group["source_file"].unique()),
                        "label": sorted(int(x) for x in group["seed_label"].unique()),
                        "has_box": sorted(bool(x) for x in group["has_human_box"].unique()),
                    }
                ),
            }
        )
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in group_rows:
        by_stratum[row["stratum"]].append(row)
    assignments: dict[str, str] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    stratum_targets: dict[str, dict[str, float]] = {}
    for stratum, groups in sorted(by_stratum.items()):
        total = sum(group["size"] for group in groups)
        counts = {split: 0 for split in RATIOS}
        targets = {split: total * RATIOS[split] for split in RATIOS}
        rng = random.Random(SEED ^ stable_int(stratum))
        ordered = sorted(groups, key=lambda group: group["group_id"])
        rng.shuffle(ordered)
        ordered.sort(key=lambda group: group["size"], reverse=True)
        for group in ordered:
            allowed = ["train", "val"] if group["forbid_test"] else list(RATIOS)
            best = min(
                allowed,
                key=lambda split: (
                    ((counts[split] + group["size"] - targets[split]) ** 2 - (counts[split] - targets[split]) ** 2) / max(targets[split], 1.0),
                    stable_int(f"{SEED}|{stratum}|{group['group_id']}|{split}"),
                ),
            )
            assignments[group["group_id"]] = best
            counts[best] += group["size"]
        stratum_counts[stratum] = counts
        stratum_targets[stratum] = targets

    desired = target_sizes(len(seed_df))
    current = Counter(assignments[group["group_id"]] for group in group_rows for _ in range(group["size"]))
    group_by_id = {group["group_id"]: group for group in group_rows}
    while any(current[split] != desired[split] for split in RATIOS):
        deficits = [split for split in RATIOS if current[split] < desired[split]]
        surpluses = [split for split in RATIOS if current[split] > desired[split]]
        candidates = []
        for group_id, old_split in assignments.items():
            group = group_by_id[group_id]
            if group["size"] != 1 or old_split not in surpluses:
                continue
            for new_split in deficits:
                if new_split == "test" and group["forbid_test"]:
                    continue
                stratum = group["stratum"]
                counts = stratum_counts[stratum]
                targets = stratum_targets[stratum]
                delta = (
                    (counts[old_split] - 1 - targets[old_split]) ** 2
                    + (counts[new_split] + 1 - targets[new_split]) ** 2
                    - (counts[old_split] - targets[old_split]) ** 2
                    - (counts[new_split] - targets[new_split]) ** 2
                )
                candidates.append((delta, stable_int(f"rebalance|{group_id}|{new_split}"), group_id, old_split, new_split, stratum))
        if not candidates:
            raise RuntimeError(f"Unable to reach exact global split sizes: current={dict(current)}, desired={desired}")
        _, _, group_id, old_split, new_split, stratum = min(candidates)
        assignments[group_id] = new_split
        current[old_split] -= 1
        current[new_split] += 1
        stratum_counts[stratum][old_split] -= 1
        stratum_counts[stratum][new_split] += 1
    seed_df["split"] = seed_df["split_group_id"].map(assignments)
    if seed_df["split"].isna().any():
        raise RuntimeError("Unassigned split rows")
    return seed_df


def attribute_qc(seed_df: pd.DataFrame, peak_df: pd.DataFrame) -> list[dict]:
    rows = []
    for entity, frame in (("seed", seed_df), ("peak_instance", peak_df)):
        for batch, group in [("all", frame), *frame.groupby("old_new_batch", sort=True)]:
            for attr in ATTRS:
                values = pd.to_numeric(group[attr], errors="coerce").replace([np.inf, -np.inf], np.nan)
                finite = values.dropna()
                rows.append(
                    {
                        "entity": entity,
                        "batch": batch,
                        "attribute": attr,
                        "rows": len(values),
                        "finite_count": len(finite),
                        "nan_count": int(values.isna().sum()),
                        "inf_count": int(np.isinf(pd.to_numeric(group[attr], errors="coerce")).sum()),
                        "min": finite.min() if len(finite) else "",
                        "q1": finite.quantile(0.25) if len(finite) else "",
                        "median": finite.median() if len(finite) else "",
                        "mean": finite.mean() if len(finite) else "",
                        "q3": finite.quantile(0.75) if len(finite) else "",
                        "max": finite.max() if len(finite) else "",
                    }
                )
    return rows


def distribution_rows(seed_df: pd.DataFrame, dimension: str) -> list[dict]:
    rows = []
    for values, group in seed_df.groupby(dimension, dropna=False, sort=True):
        rows.append(
            {
                "dimension": dimension,
                "value": values,
                "images": len(group),
                "seed_positive": int((group["seed_label"] == 1).sum()),
                "seed_negative": int((group["seed_label"] == 0).sum()),
                "with_human_box": int(group["has_human_box"].sum()),
                "true_peak_instances": int(group["true_peak_count"].sum()),
                "out_fig_instances": int(group["out_fig_count"].sum()),
            }
        )
    return rows


def leakage_audit(seed_df: pd.DataFrame) -> dict:
    checks = {}
    for field in ("image_id", "seed_id", "image", "annotation_json", "image_sha256", "split_group_id"):
        split_sets = {split: set(seed_df.loc[seed_df["split"] == split, field]) for split in RATIOS}
        intersections = {
            "train_val": sorted(split_sets["train"] & split_sets["val"]),
            "train_test": sorted(split_sets["train"] & split_sets["test"]),
            "val_test": sorted(split_sets["val"] & split_sets["test"]),
        }
        checks[field] = {key: len(value) for key, value in intersections.items()}
        if any(intersections.values()):
            raise RuntimeError(f"Split leakage in {field}: {intersections}")
    if seed_df.loc[seed_df["split"] == "test", "exclude_from_test"].any():
        raise RuntimeError("Conflict/exclusion row entered Test")
    return checks


def dataframe_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict("records")


def main(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged dataset: {output_root}")
    output_root.mkdir(parents=True)

    old_seed, old_peaks, old_images = load_old(args)
    source_hashes = dict(zip(old_seed["source_file"], old_seed["source_mzml_sha256"], strict=False))
    new_seed, new_peaks, new_images, new_jsons = load_new(args, source_hashes)
    seed_df = pd.concat([old_seed, new_seed], ignore_index=True)
    peak_df = pd.concat([old_peaks, new_peaks], ignore_index=True)
    images = {**old_images, **new_images}
    seed_df, peak_df, file_manifest = assign_ids_and_copy(seed_df, peak_df, images, new_jsons, output_root)
    seed_df, near_pairs, duplicate_groups, conflicts_and_exclusions = audit_duplicates(seed_df)
    seed_df = assign_split(seed_df)
    leakage = leakage_audit(seed_df)
    image_id_by_seed = dict(zip(seed_df["seed_id"], seed_df["image_id"], strict=True))
    peak_df["image_id"] = peak_df["seed_id"].map(image_id_by_seed)
    peak_df["split"] = peak_df["seed_id"].map(dict(zip(seed_df["seed_id"], seed_df["split"], strict=True)))

    tables = output_root / "tables"
    audits = output_root / "audits"
    splits = output_root / "splits"
    write_csv(tables / "seed_master_16attrs.csv", dataframe_records(seed_df))
    write_csv(tables / "peak_instances_master_16attrs.csv", dataframe_records(peak_df))
    write_csv(audits / "exact_duplicate_audit.csv", [], ["match_type", "match_value", "member_count", "resolution"])
    write_csv(audits / "near_duplicate_pairs.csv", near_pairs)
    write_csv(audits / "duplicate_groups.csv", duplicate_groups)
    conflict_rows = [row for row in conflicts_and_exclusions if "conflict_id" in row]
    exclusion_rows = [row for row in conflicts_and_exclusions if "conflict_id" not in row]
    write_csv(audits / "annotation_conflicts.csv", conflict_rows)
    write_csv(audits / "test_exclusions.csv", exclusion_rows)
    write_csv(audits / "attribute_qc.csv", attribute_qc(seed_df, peak_df))
    for dimension in ("domain_id", "source_file", "seed_label", "has_human_box", "old_new_batch", "difficulty_type", "split"):
        write_csv(audits / f"distribution_by_{dimension}.csv", distribution_rows(seed_df, dimension))

    manifest_columns = [
        "image_id", "seed_id", "original_seed_id", "image", "annotation_json", "seed_label", "source_file",
        "source_mzml_name", "source_mzml_sha256", "domain_id", "old_new_batch", "batch_name", "has_human_box",
        "true_peak_count", "out_fig_count", "difficulty_type", "duplicate_group_id", "split_group_id", "conflict_status",
        "exclude_from_test", "image_sha256", *ATTRS,
    ]
    for split in ("train", "val", "test"):
        frame = seed_df[seed_df["split"] == split][manifest_columns].copy()
        write_csv(splits / f"{split}.csv", dataframe_records(frame), manifest_columns)
        with (splits / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in dataframe_records(frame):
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    test_csv_sha = sha256_file(splits / "test.csv")
    test_jsonl_sha = sha256_file(splits / "test.jsonl")
    write_json(
        splits / "test_manifest_lock.json",
        {
            "status": "LOCKED",
            "created_by_seed": SEED,
            "selection_prohibition": "Test must not be used for model, epoch, threshold, preprocessing, early-stop, or hyperparameter selection.",
            "test_csv_sha256": test_csv_sha,
            "test_jsonl_sha256": test_jsonl_sha,
            "test_rows": int((seed_df["split"] == "test").sum()),
        },
    )
    (splits / "LOCKED_TEST_MANIFEST.sha256").write_text(
        f"{test_csv_sha}  test.csv\n{test_jsonl_sha}  test.jsonl\n", encoding="ascii"
    )

    source_rows = []
    for source, (domain_id, domain_name) in sorted(SOURCE_DOMAINS.items()):
        source_rows.append(
            {
                "source_file": source,
                "source_mzml_name": f"{source}.mzML",
                "domain_id": domain_id,
                "domain_name": domain_name,
                "definition_level": "independent public study or self-collected acquisition project",
                "reference_workbook": "mzML数据集公开来源核对表_简洁版.xlsx (verified 2026-08-04)",
            }
        )
    write_csv(tables / "domain_definitions.csv", source_rows)
    write_csv(audits / "file_manifest_sha256.csv", file_manifest)

    split_summary = {}
    for split, group in seed_df.groupby("split", sort=True):
        split_summary[split] = {
            "rows": len(group),
            "seed_positive": int((group["seed_label"] == 1).sum()),
            "seed_negative": int((group["seed_label"] == 0).sum()),
            "old": int((group["old_new_batch"] == "old_final_reviewed_20260725").sum()),
            "new": int((group["old_new_batch"] != "old_final_reviewed_20260725").sum()),
            "human_box_images": int(group["has_human_box"].sum()),
            "duplicate_groups": int(group.loc[group["duplicate_group_id"] != "", "duplicate_group_id"].nunique()),
        }
    audit = {
        "status": "ok",
        "frozen_seed": SEED,
        "counts": {
            "old_images": len(old_seed),
            "new_images": len(new_seed),
            "merged_images": len(seed_df),
            "old_peak_instances": len(old_peaks),
            "new_peak_instances": len(new_peaks),
            "merged_peak_instances": len(peak_df),
            "true_peak_instances": int((peak_df["label"] == "True_Peak").sum()),
            "out_fig_instances": int((peak_df["label"] == "OUT_FIG").sum()),
            "seed_positive": int((seed_df["seed_label"] == 1).sum()),
            "seed_negative": int((seed_df["seed_label"] == 0).sum()),
            "single_true_peak_images": int((seed_df["true_peak_count"] == 1).sum()),
            "multi_true_peak_images": int((seed_df["true_peak_count"] > 1).sum()),
            "no_true_peak_images": int((seed_df["true_peak_count"] == 0).sum()),
        },
        "duplicates": {
            "exact_id_groups": 0,
            "exact_image_sha256_groups": 0,
            "near_pairs_hamming_le8": len(near_pairs),
            "near_pairs_hamming_le4": sum(bool(row["high_similarity"]) for row in near_pairs),
            "near_duplicate_groups": len(duplicate_groups),
            "near_label_conflict_groups": len(conflict_rows),
            "test_exclusion_rows": len(exclusion_rows),
        },
        "near_duplicate_rules": {
            "candidate": "same source, symmetric m/z distance <=5 ppm, RT distance <=2 seconds",
            "grouping": "64-bit dHash Hamming distance <=4; connected components kept in one split",
            "review_band": "dHash Hamming distance 5-8 is audited but not grouped",
        },
        "split": split_summary,
        "leakage_intersection_counts": leakage,
        "test_lock": {"test_csv_sha256": test_csv_sha, "test_jsonl_sha256": test_jsonl_sha},
        "path_policy": "All paths stored in tables/manifests are relative to this dataset root.",
        "annotation_policy": "Old LabelMe JSON regenerated only from authoritative old-final peak jobs; new LabelMe JSON copied with shapes unchanged and imageData removed; no original protected data modified.",
    }
    write_json(audits / "merge_split_selfcheck.json", audit)

    field_rows = [
        {"field": "seed_id", "table": "seed_master/manifests", "definition": "Unique namespaced Seed/candidate ID (OLD:: or NEW::)."},
        {"field": "image_id", "table": "seed_master/peak_master/manifests", "definition": "Unique merged image ID PTL-######."},
        {"field": "domain_id", "table": "all", "definition": "Independent public study ID or self-collected acquisition project; not individual mzML."},
        {"field": "seed_label", "table": "seed_master/manifests", "definition": "Authoritative binary Seed truth label, 1=true candidate, 0=false candidate."},
        {"field": "duplicate_group_id", "table": "seed_master/manifests", "definition": "Non-empty for exact/near duplicate connected components; components cannot cross splits."},
        {"field": "exclude_from_test", "table": "seed_master/manifests", "definition": "True only for unresolved label-conflict groups; retained outside Test."},
        {"field": "SNR..EDGE", "table": "seed_master/peak_master/manifests", "definition": "Raw 16 peak attributes. Missing legacy states are preserved; train-only imputation/standardization occurs in the model pipeline."},
    ]
    write_csv(tables / "field_dictionary.csv", field_rows)
    (output_root / "README.md").write_text(
        "# PeakTruthLab final merged dataset (2026-08-14)\n\n"
        "This is a new, portable merged directory. The protected old-final dataset was not overwritten.\n\n"
        "- Seed master: `tables/seed_master_16attrs.csv`\n"
        "- Peak-instance master: `tables/peak_instances_master_16attrs.csv`\n"
        "- Locked main split: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`\n"
        "- Test lock: `splits/test_manifest_lock.json`\n"
        "- Audits and QC: `audits/`\n"
        "- Images and normalized LabelMe JSON: `images/`\n\n"
        "All 16 attributes are raw values. Missing old-final states are preserved. Imputation and standardization must be fitted on Train only.\n",
        encoding="utf-8",
    )

    top_manifest_files = [
        tables / "seed_master_16attrs.csv", tables / "peak_instances_master_16attrs.csv",
        splits / "train.csv", splits / "val.csv", splits / "test.csv", audits / "merge_split_selfcheck.json",
    ]
    final_hash_rows = [{"relative_path": path.relative_to(output_root).as_posix(), "sha256": sha256_file(path)} for path in top_manifest_files]
    write_csv(output_root / "SHA256SUMS_core.csv", final_hash_rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    repo = Path(r"D:\CODE\LipidBench")
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", default=str(repo / "PeakTruthLab/results/paper_final_reviewed_20260725/dataset_release/PeakTruthLab-dataset-v2"))
    parser.add_argument("--old-seed-jobs", default=str(repo / "PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/old_final_seed_attribute_jobs.csv"))
    parser.add_argument("--old-peak-jobs", default=str(repo / "PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/old_final_peak_attribute_jobs.csv"))
    parser.add_argument("--old-seed-calc", default=str(repo / "PeakTruthLab/results/rtx4070_oldfinal_attribute_results_20260814/old_final_seed_calculated_16.csv"))
    parser.add_argument("--old-peak-calc", default=str(repo / "PeakTruthLab/results/rtx4070_oldfinal_attribute_results_20260814/old_final_peak_calculated_16.csv"))
    parser.add_argument("--new-root", default=r"D:\CODE\downloads\PeakTruthLab_RTX4070_20260814\extracted\manual_negative_4500_v2_staging")
    parser.add_argument("--output-root", default=str(repo / "PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
