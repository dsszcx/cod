"""Separate, official-network inference / protocol verification. Never trains."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import evaluate, prediction_to_png

COMMIT = '5b9844c94d10a03b815c7c7039c194166a9a8ba3'
REFERENCE = {
    'CAMO': dict(sm=.820, mean_e=.882, wfm=.743, mae=.070),
    'COD10K': dict(sm=.815, mean_e=.887, wfm=.680, mae=.037),
    'CHAMELEON': dict(sm=.888, mean_e=.942, wfm=.816, mae=.030),
}
COUNTS = {'CAMO': 250, 'COD10K': 2026, 'CHAMELEON': 76}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pairs(root, dataset):
    images = sorted(p for p in (root/dataset/'Imgs').iterdir()
                    if p.suffix.lower() in {'.jpg', '.png'})
    if dataset == 'COD10K':
        images = [p for p in images if '-CAM-' in p.name]
    items = [(p, root/dataset/'GT'/(p.stem+'.png')) for p in images]
    if len(items) != COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected {COUNTS[dataset]} images, got {len(items)}')
    if len({p.stem for p, _ in items}) != len(items):
        raise ValueError('Duplicate image stems')
    for p, gt in items:
        if not gt.is_file():
            raise FileNotFoundError(gt)
        with Image.open(gt) as mask, Image.open(p) as image:
            if mask.size != image.size:
                raise ValueError(f'Image/GT size mismatch: {p.name}')
            arr = np.array(mask.convert('L'))
            if not np.isin(arr, [0, 255]).all() or not (arr > 128).any():
                raise ValueError(f'Nonbinary or empty benchmark GT: {gt}')
    return items


def evaluate_file(item):
    name, prediction, target = item
    with Image.open(prediction) as pred, Image.open(target) as mask:
        return dict(name=name, **evaluate(np.array(pred.convert('L')), np.array(mask.convert('L'))))


def infer(args, all_pairs):
    if not args.weights or not args.weights.is_file():
        raise FileNotFoundError('Official Net_epoch_best.pth is required; no substitute weights')
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f'Unexpected upstream commit: {commit}')
    changes = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only', 'HEAD', '--',
                                       'lib', 'utils/data_val.py', 'MyTesting.py', 'eval'], text=True)
    if changes.strip():
        raise ValueError('Upstream inference/evaluation source has local changes')
    sys.path.insert(0, str(upstream))
    from lib.Network_Res2Net_GRA_NCD import Network
    import lib.Network_Res2Net_GRA_NCD as source
    if upstream not in Path(source.__file__).resolve().parents:
        raise RuntimeError('Imported project network instead of official network')
    device = torch.device(args.device)
    model = Network(imagenet_pretrained=False)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    transform = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    metadata = dict(upstream_commit=commit, weights_sha256=sha256(args.weights),
                    weights_name=args.weights.name, torch=torch.__version__,
                    source_url='https://github.com/GewelsJI/SINet-V2',
                    weight_url='https://drive.google.com/file/d/1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR/view',
                    size=352, output='last of four outputs', amp=False,
                    png_writer='SciPy misc.imsave bytescale-compatible uint8 rounding',
                    architecture_source=str(source.__file__))
    (args.output/'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with torch.inference_mode():
        for dataset, items in all_pairs.items():
            dest = args.output/'predictions'/dataset
            dest.mkdir(parents=True, exist_ok=True)
            if any(dest.glob('*.png')):
                raise FileExistsError(f'Existing predictions: {dest}; choose a new output directory')
            for index, (path, gt) in enumerate(items, 1):
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                    tensor = transform(rgb).unsqueeze(0).to(device)
                    shape = (rgb.height, rgb.width)
                logits = model(tensor)[3]
                prob = F.interpolate(logits, size=shape, mode='bilinear', align_corners=False)
                prob = prob.sigmoid().cpu().numpy().squeeze()
                png = prediction_to_png(prob)
                Image.fromarray(png).save(dest/(path.stem+'.png'))
                if index == 1 or index % 100 == 0 or index == len(items):
                    print(f'INFER {dataset} {index}/{len(items)}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['check', 'infer', 'evaluate', 'all'], default='check')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=Path('references/upstream/SINet-V2'))
    p.add_argument('--weights', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=list(COUNTS), default=['CAMO', 'COD10K'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    all_pairs = {d: pairs(args.data_root, d) for d in args.datasets}
    print('DATA_CHECK_PASSED '+json.dumps({k: len(v) for k, v in all_pairs.items()}), flush=True)
    if args.phase == 'check':
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in ['infer', 'all']:
        infer(args, all_pairs)
    if args.phase in ['evaluate', 'all']:
        results = dict(protocol='Python port of upstream MATLAB protocol; NOT a MATLAB execution',
                       caveat='SciPy EDT nearest-foreground ties may differ from MATLAB bwdist',
                       upstream_commit=COMMIT, datasets={})
        for dataset, items in all_pairs.items():
            folder = args.output/'predictions'/dataset
            expected = {p.stem+'.png' for p, _ in items}
            actual = {p.name for p in folder.glob('*.png')}
            if actual != expected:
                raise ValueError(f'{dataset} prediction coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
            rows = []
            jobs = [(image.stem, folder/(image.stem+'.png'), gt) for image, gt in items]
            # Spawn: evaluation workers never inherit a live CUDA context.
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
                for index, row in enumerate(pool.map(evaluate_file, jobs, chunksize=4), 1):
                    rows.append(row)
                    if index % 200 == 0 or index == len(items):
                        print(f'EVAL {dataset} {index}/{len(items)}', flush=True)
            means = {k: float(np.mean([r[k] for r in rows])) for k in ['sm', 'mean_e', 'wfm', 'mae']}
            results['datasets'][dataset] = dict(count=len(rows), metrics=means,
                paper_reference=REFERENCE[dataset], delta={k: means[k]-REFERENCE[dataset][k] for k in means})
            (args.output/(dataset+'_per_image.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(dataset+' '+json.dumps(results['datasets'][dataset]), flush=True)
        (args.output/'metrics_python.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print('OFFICIAL_SINETV2_PYTHON_PROTOCOL_CHECK_DONE', flush=True)


if __name__ == '__main__':
    main()
