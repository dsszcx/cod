"""Fixed full-data RGB development training; never opens a test manifest."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataloader_relation import RelationDataset, collate_relation_batch
from tools.e0e4 import common as C
from tools.semantic_prior.training_core import build_model, deep_loss, seed_all, state_digest
from tools.semantic_prior.evaluation_runtime import configure_evaluation
from tools.semantic_prior.full_training_state import signature, make_scheduler, save_checkpoint, latest_checkpoint, restore
from tools.semantic_prior.pilot_metrics import raw_metrics, require_primary_metrics
from tools.semantic_prior.prepare_cam_only import category
from tools.official_sinetv2.metrics import evaluate as standard_metrics, prediction_to_png


def score_probability(job):
    row, probability = job
    with Image.open(row['gt_path']) as image:
        gt = np.array(image.convert('L'))
    if not np.isin(gt, [0, 255]).all():
        raise ValueError('Nonbinary validation GT: '+row['image_id'])
    raw = raw_metrics(probability, gt.astype(np.float32)/255)
    standard = standard_metrics(prediction_to_png(probability), gt)
    require_primary_metrics(raw)
    require_primary_metrics(standard)
    return dict(image_id=row['image_id'], dataset=category(row), raw=raw, png_standard=standard)


def average_metrics(rows):
    answer = {}
    for protocol in ('raw', 'png_standard'):
        answer[protocol] = {}
        for name in ('all', 'CAMO', 'COD10K-CAM'):
            subset = [r for r in rows if name == 'all' or r['dataset'] == name]
            if not subset:
                raise ValueError('Missing validation group: '+name)
            answer[protocol][name] = dict(n_images=len(subset), **{
                k: float(np.mean([r[protocol][k] for r in subset]))
                for k in ('mae', 'sm', 'mean_e', 'wfm')})
    return answer


def validate(model, dataset, pool, device):
    model.eval()

    def jobs():
        with torch.inference_mode():
            for index, row in enumerate(dataset.rows):
                item = dataset[index]
                batch = collate_relation_batch([item])
                output = model(batch['inputs']['rgb'].to(device))[-1]
                output = F.interpolate(output.float(), size=item['native_hw'], mode='bilinear', align_corners=False)
                probability = output.sigmoid()[0, 0].cpu().numpy()
                yield row, probability

    rows = list(pool.map(score_probability, jobs(), chunksize=4))
    if len(rows) != len(dataset):
        raise ValueError('Incomplete validation')
    return average_metrics(rows), rows


def write_summary(root, state, status):
    C.write_json(str(root/'SUMMARY.json'), dict(status=status,
        completed_epochs=state['epoch'], planned_epochs=state['config']['epochs'],
        optimizer_steps=state['optimizer_steps'], best_epoch=state['best_epoch'],
        best_raw_val_mae=state['best'], latest_metrics=state['metrics'],
        n_train=3652, n_val=383, test_set_used=False,
        baseline='Project PVTv2 RGB, not official Res2Net SINet-V2',
        elapsed_training_s=state['elapsed_s']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = C.load_config(args.config)
    if cfg['data_scope'] != 'cam_only' or cfg['checkpoint_selection'] != 'raw_val_mae':
        raise ValueError('Unexpected scope or checkpoint selector')
    runtime = configure_evaluation(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA device is required')
    device = torch.device('cuda:0')
    manifests = Path(cfg['paths']['manifests'])
    paths = [manifests/(s+'.csv') for s in ('train', 'val')]
    rows = [C.read_csv(str(p)) for p in paths]
    expected = [cfg['expected_train_images'], cfg['expected_val_images']]
    if [len(r) for r in rows] != expected:
        raise ValueError('Training/validation counts changed')
    if any(category(r) == 'NonCAM' for split in rows for r in split):
        raise ValueError('NonCAM in a CAM-only training run')
    if {r['image_sha256'] for r in rows[0]} & {r['image_sha256'] for r in rows[1]}:
        raise ValueError('Train/val image overlap')
    hashes = {p.name: C.sha256_file(str(p)) for p in paths}
    hashes['imagenet'] = C.sha256_file(cfg['paths']['imagenet'])
    run_signature = signature(cfg, hashes)
    root = Path(cfg['paths']['out_dir'])
    if args.resume:
        if not root.is_dir():
            raise FileNotFoundError(root)
    else:
        root.mkdir(parents=True, exist_ok=False)
        C.write_json(str(root/'CONFIG.json'), {k:v for k,v in cfg.items() if not k.startswith('_')})
    model, initialization = build_model('rgb', cfg['seed'], cfg['paths']['imagenet'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = make_scheduler(optimizer, cfg)
    code = ('Network_PVTv2.py', 'dataloader_relation.py', 'lib/pvtv2.py',
            'tools/semantic_prior/training_core.py', 'tools/semantic_prior/train_rgb_full.py',
            'tools/semantic_prior/full_training_state.py', 'tools/semantic_prior/pilot_metrics.py',
            'tools/official_sinetv2/metrics.py')
    project = Path(__file__).resolve().parents[2]
    provenance = dict(run_signature=run_signature, initialization=initialization,
        input_hashes=hashes, runtime=runtime, torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0), test_set_used=False,
        official_sinet_weights_used=False, acceptance_checkpoint_used=False,
        amp=False, data_workers=0, metric_workers=cfg['metric_workers'],
        epoch_seed_policy='seed_all(seed+zero_based_epoch); stateless crop/flip; seeded loader generator',
        resume_scope='completed epoch; an interrupted epoch is replayed, not skipped',
        code_sha256={name:C.sha256_file(str(project/name)) for name in code})
    start_epoch, steps, best, best_epoch, elapsed_before = 0, 0, float('inf'), 0, 0.
    history = []
    if args.resume:
        checkpoint_path, saved = latest_checkpoint(root)
        restore(saved, model, optimizer, scheduler, run_signature)
        start_epoch, steps = saved['epoch'], saved['optimizer_steps']
        best, best_epoch = saved['best'], saved['best_epoch']
        history, elapsed_before = saved['history'], saved['elapsed_s']
        print('RESUMED', checkpoint_path.name, 'completed_epochs', start_epoch, flush=True)
        C.write_json(str(root/('RESUME_PROVENANCE_%03d.json' % start_epoch)), provenance)
        del saved
    else:
        C.write_json(str(root/'PROVENANCE.json'), provenance)
    train = RelationDataset(paths[0], 'train', 'rgb', cfg['input_size'], cfg['seed'])
    val = RelationDataset(paths[1], 'val', 'rgb', cfg['input_size'], cfg['seed'])
    print('RGB_FULL_START', json.dumps(dict(epochs=cfg['epochs'], start_epoch=start_epoch,
          n_train=len(train), n_val=len(val), batch=cfg['batch_size'], runtime=runtime)), flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['metric_workers'], mp_context=get_context('spawn')) as pool:
        for epoch in range(start_epoch, cfg['epochs']):
            train.set_epoch(epoch)
            generator = torch.Generator().manual_seed(cfg['seed']+epoch)
            loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, generator=generator,
                num_workers=0, drop_last=False, collate_fn=collate_relation_batch)
            seed_all(cfg['seed']+epoch)
            model.train()
            lr = optimizer.param_groups[0]['lr']
            losses, order = [], []
            for index, batch in enumerate(loader, 1):
                order.extend(batch['image_ids'])
                optimizer.zero_grad(set_to_none=True)
                output = model(batch['inputs']['rgb'].to(device))
                loss = deep_loss(output, batch['gt'].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient norm')
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
                if index == 1 or index % 50 == 0 or index == len(loader):
                    print('TRAIN', epoch+1, '/', cfg['epochs'], 'step', index, '/', len(loader),
                          'lr', lr, 'loss', losses[-1], flush=True)
            metrics, per_image = validate(model, val, pool, device)
            score = metrics['raw']['all']['mae']
            improved = score < best
            if improved:
                best, best_epoch = score, epoch+1
            scheduler.step()
            record = dict(epoch=epoch+1, learning_rate=lr, next_learning_rate=optimizer.param_groups[0]['lr'],
                mean_training_loss=float(np.mean(losses)), optimizer_steps=steps, metrics=metrics)
            history.append(record)
            C.write_json(str(root/('epoch_%03d.json' % (epoch+1))), dict(**record, sample_order=order, per_image=per_image))
            metadata = dict(epoch=epoch+1, optimizer_steps=steps, best=best, best_epoch=best_epoch,
                history=history, metrics=metrics, initialization=initialization,
                config={k:v for k,v in cfg.items() if not k.startswith('_')},
                run_signature=run_signature, elapsed_s=elapsed_before+time.monotonic()-started)
            if improved:
                save_checkpoint(root/'best.pth', model, optimizer, scheduler, **metadata)
            save_checkpoint(root/'last.pth', model, optimizer, scheduler, **metadata)
            C.write_json(str(root/'HISTORY.json'), history)
            write_summary(root, metadata, 'RUNNING')
            print('EPOCH_DONE', json.dumps(record), 'best_epoch', best_epoch, flush=True)
    saved_best = torch.load(root/'best.pth', map_location='cpu', weights_only=True)
    C.write_json(str(root/'RESULTS.json'), dict(status='DONE', selected_epoch=saved_best['epoch'],
        metrics=saved_best['metrics'], checkpoint_selection='raw_val_mae',
        n_train=len(train), n_val=len(val), planned_epochs=cfg['epochs'],
        test_set_used=False, claim='Full-length fixed RGB development baseline; not an official test result'))
    completed = latest_checkpoint(root)[1]
    write_summary(root, completed, 'DONE')
    print('RGB_CAM_FULL_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()
