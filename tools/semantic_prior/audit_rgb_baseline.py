"""Replay the fixed RGB best checkpoint, then benchmark it once without tuning."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.training_core import build_model
from tools.semantic_prior.evaluation_runtime import configure_evaluation
from tools.semantic_prior.prepare_cam_only import category
from tools.semantic_prior.train_rgb_full import average_metrics
from tools.semantic_prior.pilot_metrics import raw_metrics
from tools.official_sinetv2.metrics import evaluate, prediction_to_png
from lib.region_relations import transform_region_sample

KEYS = ('mae', 'sm', 'mean_e', 'wfm')


def rgb_input(path, size):
    """Exactly the existing project's validation transform, without reading GT."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), np.float32).transpose(2, 0, 1)/255
    h, w = rgb.shape[1:]
    sample = dict(rgb=rgb, masks=np.zeros((0,h,w), np.uint8),
                  role_hint=np.zeros((0,4), np.float32), role_hint_valid=np.zeros(0, bool))
    sample = transform_region_sample(sample, output_hw=(size, size))
    tensor = (sample['rgb']-np.array([.485,.456,.406],np.float32)[:,None,None]) / np.array([.229,.224,.225],np.float32)[:,None,None]
    return torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0), (h,w)


def score(job):
    row, probability, prediction = job
    with Image.open(row['gt_path']) as mask:
        gt = np.array(mask.convert('L'))
    with Image.open(prediction) as image:
        png = np.array(image.convert('L'))
    if not np.isin(gt, [0,255]).all():
        raise ValueError('Nonbinary evaluation GT')
    return dict(image_id=row['image_id'], dataset=category(row),
        raw=raw_metrics(probability, gt.astype(np.float32)/255), png_standard=evaluate(png, gt))


def replay_differences(actual, expected):
    previous = {r['image_id']:r for r in expected}
    if {r['image_id'] for r in actual} != set(previous):
        raise ValueError('Validation replay coverage changed')
    differences = {}
    for protocol in ('raw','png_standard'):
        for key in KEYS:
            differences[protocol+'/'+key] = max(abs(r[protocol][key]-previous[r['image_id']][protocol][key]) for r in actual)
    # Exact replay is expected with the same source/runtime/device settings.
    if any(v != 0 for v in differences.values()):
        raise ValueError('Validation replay differs; do not run test: '+str(differences))
    return differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', required=True, choices=['validate','test'])
    args = p.parse_args()
    train_root, output = args.training, args.output
    summary = C.read_json(str(train_root/'SUMMARY.json'))
    selection = C.read_json(str(train_root/'RESULTS.json'))
    cfg = C.read_json(str(train_root/'CONFIG.json'))
    provenance = C.read_json(str(train_root/'PROVENANCE.json'))
    history = C.read_json(str(train_root/'HISTORY.json'))
    if summary['status'] != 'DONE' or summary['completed_epochs'] != 80 or summary['optimizer_steps'] != 36560:
        raise ValueError('Training is incomplete')
    selected = min(history, key=lambda r:r['metrics']['raw']['all']['mae'])['epoch']
    if selection['selected_epoch'] != selected or selection['checkpoint_selection'] != 'raw_val_mae':
        raise ValueError('Checkpoint selector changed')
    project = Path(__file__).resolve().parents[2]
    for name, digest in provenance['code_sha256'].items():
        if C.sha256_file(str(project/name)) != digest:
            raise ValueError('Frozen training source differs: '+name)
    checkpoint = train_root/'best.pth'
    weight_hash = C.sha256_file(str(checkpoint))
    if args.phase == 'validate':
        output.mkdir(parents=True, exist_ok=False)
        C.write_json(str(output/'LOCK.json'), dict(selected_epoch=selected,
            checkpoint_sha256=weight_hash, checkpoint=str(checkpoint),
            selection='Previously fixed raw development-val MAE; not test selection',
            source_training=str(train_root), data_version='Existing CAM-only v1, unchanged',
            no_hyperparameter_or_checkpoint_search=True))
    else:
        replay = C.read_json(str(output/'VAL_REPLAY.json'))
        if replay is None or replay['status'] != 'PASS' or replay['checkpoint_sha256'] != weight_hash:
            raise ValueError('Exact validation replay must pass before test')
    name = 'VAL_REPLAY' if args.phase == 'validate' else 'TEST_RESULTS'
    if (output/(name+'.json')).exists():
        raise FileExistsError('Phase already has results: '+name)
    runtime = configure_evaluation(cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['epoch'] != selected:
        raise ValueError('best.pth epoch differs from frozen selection')
    model, initialization = build_model('rgb',cfg['seed'],cfg['paths']['imagenet'])
    if initialization['rgb_initial_sha256'] != saved['initialization']['rgb_initial_sha256']:
        raise ValueError('Initialization source differs')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.cuda().eval()
    split = 'val' if args.phase == 'validate' else 'test'
    manifest = Path(cfg['paths']['manifests'])/(split+'.csv')
    rows = C.read_csv(str(manifest))
    counts = dict(Counter(category(row) for row in rows))
    expected = {'CAMO':96,'COD10K-CAM':287} if split == 'val' else {'CAMO':250,'COD10K-CAM':2026}
    if counts != expected or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected fixed evaluation set: '+str(counts))
    predictions = output/('validation_predictions' if split == 'val' else 'test_predictions')
    predictions.mkdir(exist_ok=False)

    def jobs():
        with torch.inference_mode():
            for index,row in enumerate(rows,1):
                if C.sha256_file(row['image_path']) != row['image_sha256']:
                    raise ValueError('Evaluation image content changed')
                tensor, shape = rgb_input(row['image_path'],cfg['input_size'])
                logits = model(tensor.cuda())[-1]
                probability = F.interpolate(logits.float(),size=shape,mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
                path = predictions/(row['image_id'].replace('/','__')+'.png')
                Image.fromarray(prediction_to_png(probability)).save(path)
                if index == 1 or index%200 == 0 or index == len(rows):
                    print(args.phase.upper(),'INFER',index,'/',len(rows),flush=True)
                yield row,probability,str(path)

    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        per_image = list(pool.map(score,jobs(),chunksize=4))
    if C.sha256_file(str(checkpoint)) != weight_hash:
        raise ValueError('Selected checkpoint changed during inference')
    metrics = average_metrics(per_image)
    result = dict(status='PASS' if split == 'val' else 'DONE', selected_epoch=selected,
        checkpoint_sha256=weight_hash,n_images=len(rows),counts=counts,metrics=metrics,
        runtime=runtime,manifest_sha256=C.sha256_file(str(manifest)),
        protocol='Same PNG bytescale and Python standard metric port as official-weight verification; not MATLAB execution',
        data_version='Existing CAM-only v1, unchanged', test_history='Public benchmark has previous project development exposure; not a pristine confirmatory set')
    if split == 'val':
        old = C.read_json(str(train_root/('epoch_%03d.json'%selected)))['per_image']
        result['maximum_per_image_metric_difference'] = replay_differences(per_image,old)
    C.write_json(str(output/(name+'_PER_IMAGE.json')),per_image)
    C.write_json(str(output/(name+'.json')),result)
    print(name,metrics,flush=True)
    print('RGB_BASELINE_'+('VALIDATION_REPLAY_PASSED' if split=='val' else 'STANDARD_TEST_DONE'),flush=True)


if __name__ == '__main__':
    main()
