"""Portable replay of 16 files x 1,000 targets x seven actual RT translations."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
LOCKED_CHECKPOINT_SHA256 = '374f81705352875890ee26f8daec8334162644e209ddec60f5de51bbf3b6c695'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prediction_signature(device_name):
    import torch
    import torchvision
    import numpy
    import pyopenms
    import matplotlib
    from matplotlib import font_manager
    import runtime
    device = runtime.resolve_device(device_name)
    files = sorted(SCRIPT_DIR.glob('*.py')) + [
        PROJECT_ROOT / 'lipidbench/utils/peak_attributes.py',
        PROJECT_ROOT / 'lipidbench/utils/plot_eic.py',
        PROJECT_ROOT / 'lipidbench/models/peak_multitask_rcnn.py',
        PROJECT_ROOT / 'PeakTruthLab/scripts/annotation/run_annotation_standardization_pilot.py',
    ]
    return dict(checkpoint_sha256=sha256(runtime.CHECKPOINT), device=str(device),
                gpu_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                numpy_version=numpy.__version__, pyopenms_version=pyopenms.__version__,
                matplotlib_version=matplotlib.__version__,
                renderer_font=font_manager.findfont('Arial', fallback_to_default=False),
                source_sha256={str(f.relative_to(PROJECT_ROOT)): sha256(f) for f in files})


def check_prediction_environment(device_name):
    import runtime
    signature = prediction_signature(device_name)
    if signature['checkpoint_sha256'] != LOCKED_CHECKPOINT_SHA256:
        raise ValueError('Historical checkpoint changed.')
    preflight_path = runtime.OUTPUT / 'hardware_preflight.json'
    if not preflight_path.exists():
        raise RuntimeError('Run preflight successfully before prediction.')
    recorded = json.loads(preflight_path.read_text(encoding='utf8'))
    if recorded.get('prediction_signature') != signature:
        raise RuntimeError('Software, device or source changed since preflight. Use a fresh work directory for a new environment.')
    lock = runtime.WORK / 'prediction_environment.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf8')) != signature:
        raise RuntimeError('Cached predictions belong to a different environment. Use a fresh work directory.')
    if not lock.exists():
        lock.write_text(json.dumps(signature, indent=2), encoding='utf8')


def initialize(args):
    import pandas as pd
    work = args.work_dir.resolve()
    if (work / 'runtime_config.json').exists():
        raise FileExistsError('Work directory already initialized. Resume with a stage command, or choose a new work directory.')
    bundle = args.bundle.resolve()
    source = bundle / 'rt_shift16'
    manifest = pd.read_csv(PROJECT_ROOT / 'PeakTruthLab/final_delivery/rt_validation_20260930/mzml16_manifest.csv')
    # Preserve an ASCII junction spelling for pyOpenMS builds that cannot open
    # a Unicode target path on Windows. Hash checks still validate the bytes.
    mzml_dir = args.mzml_dir.absolute()
    samples = []
    for row in manifest.itertuples():
        path = mzml_dir / row.filename
        if sha256(path) != row.sha256:
            raise ValueError(f'Original mzML hash differs: {path}')
        samples.append(dict(sample=row.sample, source_path=str(path), mzml_path=str(path), batch=row.batch))
    checkpoint = args.checkpoint.resolve()
    if sha256(checkpoint) != LOCKED_CHECKPOINT_SHA256:
        raise ValueError('Use the exact historical image-only checkpoint from the RT validation Release.')
    rscript = args.rscript or shutil.which('Rscript')
    if not rscript:
        raise FileNotFoundError('Supply --rscript or add Rscript to PATH.')
    work.mkdir(parents=True, exist_ok=True)
    (work / 'results').mkdir(exist_ok=True)
    for name in ['reference_feature_table_frozen.csv', 'baseline_references_frozen.csv', 'freeze.json']:
        shutil.copyfile(source / name, work / name)
        shutil.copyfile(source / name, work / 'results' / name)
    # Replaying the archived 1,000 targets; baseline never reselects the cohort.
    shutil.copyfile(source / 'reference_feature_table_frozen.csv', work / 'candidate_feature_table.csv')
    pd.DataFrame(samples).to_csv(work / 'samples16.csv', index=False)
    freeze = json.loads((work / 'freeze.json').read_text(encoding='utf8'))
    for name, key in [('reference_feature_table_frozen.csv', 'reference_feature_table_sha256'),
                      ('baseline_references_frozen.csv', 'baseline_references_sha256')]:
        if sha256(work / name) != freeze[key]:
            raise ValueError(f'Frozen reference hash differs: {name}')
    config = dict(bundle=str(bundle), checkpoint=str(checkpoint), checkpoint_sha256=LOCKED_CHECKPOINT_SHA256,
                  rscript=str(Path(rscript).resolve()), initialized_at_utc=datetime.now(timezone.utc).isoformat(),
                  reference_policy='replay original fixed priors and historical own-method baseline areas; no reselection',
                  sample_manifest_relocation='paths relocated; input mzML SHA-256 checked against archived manifest')
    (work / 'runtime_config.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    print(f'Initialized 16 files and 1,000 frozen targets: {work}', flush=True)


def preflight(args):
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    from rtshift_trace import extract_batch
    from torchvision.ops import nms, roi_align
    from torchvision.transforms.functional import to_tensor
    from PIL import Image
    import runtime
    from recovery_helpers import load_model, load_ms1_spectra, _extract_trace, valid_candidates
    from rtshift_render import render
    if sha256(runtime.CHECKPOINT) != LOCKED_CHECKPOINT_SHA256:
        raise ValueError('Checkpoint changed since initialization.')
    device = runtime.resolve_device(args.device)
    boxes = torch.tensor([[0., 0., 4., 4.], [1., 1., 3., 3.]], device=device)
    nms_result = nms(boxes, torch.tensor([.9, .8], device=device), .5)
    roi = roi_align(torch.ones((1, 2, 8, 8), device=device), [boxes], output_size=(2, 2))
    if roi.shape != (2, 2, 2, 2) or not torch.isfinite(roi).all():
        raise RuntimeError('RoIAlign preflight failed.')
    version = subprocess.run([str(runtime.RSCRIPT), '-e', "suppressPackageStartupMessages(library(xcms));suppressPackageStartupMessages(library(MSnbase));cat(as.character(packageVersion('xcms')))"],
                             check=True, capture_output=True, text=True).stdout.strip()
    if version != '4.10.1':
        raise RuntimeError(f'XCMS {version} differs from locked 4.10.1. Use the archived R runtime for an exact replay.')
    sample = pd.read_csv(runtime.WORK / 'samples16.csv').iloc[0]
    target = pd.read_csv(runtime.WORK / 'reference_feature_table_frozen.csv').iloc[0]
    spectra = load_ms1_spectra(Path(sample.source_path), backend='pyopenms')
    rt, y, _ = _extract_trace(spectra, float(target.reference_mz), 10., 'ppm', 'nearest')
    batch_rt, batch_y = extract_batch(spectra, np.array([float(target.reference_mz)]))
    if not np.array_equal(rt, batch_rt) or not np.array_equal(y, batch_y[0]):
        raise RuntimeError('Batch EIC extractor differs from the historical nearest-centroid trace.')
    prior = float(target.reference_rt)
    mask = (rt >= prior - 1) & (rt <= prior + 1)
    image_path, slope, intercept = render((rt[mask], y[mask], prior, str(runtime.WORK / 'preflight'), 'first_target'))
    model, info = load_model(runtime.CHECKPOINT, device)
    if info['fusion_mode'] != 'image_only':
        raise ValueError('Historical RT checkpoint must be image_only.')
    with torch.inference_mode():
        detection = model.detector([to_tensor(Image.open(image_path).convert('RGB')).to(device)])[0]
    cs = valid_candidates(detection, rt[mask], y[mask], rt, y, prior, slope, intercept)
    cs.sort(key=lambda c: (round(abs(c['pred_apex_rt'] - prior), 3), -c['pred_score']))
    if not cs:
        raise RuntimeError('First frozen target has no valid model candidate in preflight.')
    # Reference access follows inference, only to diagnose replay compatibility.
    refs = pd.read_csv(runtime.WORK / 'baseline_references_frozen.csv')
    ref = refs[refs.method.eq('model') & refs.feature_id.eq(target.feature_id) & refs['sample'].eq(sample['sample'])].iloc[0]
    valid = abs(cs[0]['pred_apex_rt'] - ref.common_baseline_apex_rt) <= 10 / 60 and cs[0]['pred_left'] <= ref.common_baseline_apex_rt <= cs[0]['pred_right']
    if not valid:
        raise RuntimeError('First target identity differs from archived baseline; inspect preflight image and software versions.')
    report = dict(device=str(device), gpu_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  torch_version=torch.__version__, torchvision_version=torchvision.__version__, hip_version=torch.version.hip,
                  cuda_version=torch.version.cuda, python_version=sys.version, numpy_version=np.__version__,
                  xcms_version=version, nms_pass=True, roi_align_pass=True, exact_checkpoint_hash_pass=True,
                  model_info=info, first_target=dict(feature_id=target.feature_id, sample=sample['sample'], selected=cs[0]),
                  prediction_signature=prediction_signature(args.device),
                  created_at_utc=datetime.now(timezone.utc).isoformat())
    (runtime.OUTPUT / 'hardware_preflight.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


def baseline_check():
    import numpy as np
    import pandas as pd
    import runtime
    refs = pd.read_csv(runtime.WORK / 'baseline_references_frozen.csv')
    records = []
    for method in ['xcms', 'model']:
        for sample in pd.read_csv(runtime.WORK / 'samples16.csv')['sample']:
            d = pd.read_csv(runtime.WORK / f'baseline_{method}' / f'{sample}.csv')
            records.append(d.assign(method=method))
    d = pd.concat(records).merge(refs, on=['feature_id', 'sample', 'method'], validate='one_to_one')
    assert len(d) == 32000
    expected = d.common_baseline_apex_rt
    d['baseline_identity_pass'] = d.pred_found.eq(1) & (d.pred_apex_rt - expected).abs().le(10 / 60) & d.pred_left.le(expected) & d.pred_right.ge(expected)
    d['baseline_replay_area_error_pct'] = (d.pred_area / d.baseline_area - 1).abs() * 100
    d['baseline_replay_quantification_pass'] = d.baseline_identity_pass & d.baseline_replay_area_error_pct.lt(20)
    d.to_csv(runtime.OUTPUT / 'baseline_hardware_comparison.csv', index=False)
    if not d.baseline_replay_quantification_pass.all():
        raise RuntimeError('Baseline differs from archived targets/areas; inspect baseline_hardware_comparison.csv before running shifts.')
    print('All 32,000 baseline method x file x feature records pass replay compatibility.', flush=True)


def compare_summary():
    import pandas as pd
    import runtime
    old = pd.read_csv(Path(runtime.CONFIG['bundle']) / 'rt_shift16/summary_by_method_shift.csv')
    new = pd.read_csv(runtime.OUTPUT / 'summary_by_method_shift.csv')
    joined = new.merge(old, on=['method', 'rt_shift'], suffixes=('_replay', '_rtx4070'), validate='one_to_one')
    for col in ['recall', 'accurate_quantification_rate']:
        joined[f'{col}_change_pp'] = (joined[f'{col}_replay'] - joined[f'{col}_rtx4070']) * 100
    joined['median_area_error_change_pp'] = joined.area_error_median_pct_replay - joined.area_error_median_pct_rtx4070
    joined.to_csv(runtime.OUTPUT / 'hardware_summary_comparison.csv', index=False)
    print(joined[['method', 'rt_shift', 'recall_change_pp', 'accurate_quantification_rate_change_pp']].to_string(index=False), flush=True)


def call_script(name, *options):
    subprocess.run([sys.executable, str(SCRIPT_DIR / name), *map(str, options)], check=True, env=os.environ.copy())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['init', 'preflight', 'shift', 'baseline', 'baseline-check', 'predict', 'evaluate', 'all'])
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--mzml-dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--rscript')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        p.error('--workers and --batch-size must be positive')
    os.environ['CHROMAPEAK_PROJECT_ROOT'] = str(PROJECT_ROOT)
    os.environ['CHROMAPEAK_RT_WORKDIR'] = str(args.work_dir.resolve())
    if args.stage == 'init':
        if not all([args.bundle, args.mzml_dir, args.checkpoint]):
            p.error('init requires --bundle, --mzml-dir and --checkpoint')
        initialize(args)
        return
    if not (args.work_dir / 'runtime_config.json').exists():
        p.error('Initialize this work directory first.')
    stages = ['preflight', 'shift', 'baseline', 'predict', 'evaluate'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        print(f'RT replay stage: {stage}', flush=True)
        if stage == 'preflight':
            preflight(args)
        elif stage == 'shift':
            call_script('rtshift_mzml.py')
            call_script('rtshift_audit_mzml.py')
        elif stage in ['baseline', 'predict']:
            check_prediction_environment(args.device)
            if stage == 'predict':
                baseline_check()
            options = ['--baseline'] if stage == 'baseline' else []
            call_script('rtshift_run_xcms.py', *options, '--workers', args.workers)
            call_script('rtshift_model.py', *options, '--device', args.device, '--workers', args.workers, '--batch-size', args.batch_size)
            if stage == 'baseline':
                baseline_check()
        elif stage == 'baseline-check':
            baseline_check()
        elif stage == 'evaluate':
            call_script('rtshift_evaluate.py')
            call_script('rtshift_xcms_translation_audit.py')
            call_script('rtshift_diagnostics.py')
            compare_summary()


if __name__ == '__main__':
    main()
