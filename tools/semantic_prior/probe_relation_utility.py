"""Exploratory directed-edge removal; utility is NOT a semantic relation label."""
import argparse
import gc
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.training_core import build_model
from tools.semantic_prior.candidate_supervision import candidate_targets
from dataloader_relation import RelationDataset, collate_relation_batch
from tools.semantic_prior.evaluation_runtime import configure_evaluation


def edge_indices(inputs, effective_valid, max_edges):
    """All eligible ordered pairs, fixed lexicographic prefix; NO GT/ranking."""
    n = effective_valid.shape[1]
    pairs = (inputs['relation_valid'][0].bool() & effective_valid[0, :, None]
             & effective_valid[0, None, :] & ~torch.eye(n, dtype=torch.bool, device=effective_valid.device))
    return pairs.nonzero(as_tuple=False).cpu().tolist()[:max_edges]


def role(inputs, index):
    if not inputs['role_hint_valid'][0, index]:
        return 'unavailable'
    return ('whole', 'part', 'context', 'unknown')[int(inputs['role_hint'][0, index].argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--max-edges', type=int, default=12)
    args = ap.parse_args()
    if args.max_edges < 1:
        raise ValueError('Positive edge cap required')
    plan_path = resolve(args.plan)
    root = plan_path.parent
    destination = root / 'RELATION_EDGE_UTILITY.json'
    if destination.exists():
        raise FileExistsError(str(destination))
    plan = C.read_json(str(plan_path))
    reports, runtime_settings = [], {}
    for job in plan['jobs']:
        cfg = C.load_config(job['config'])
        runtime_settings[Path(job['out_dir']).name] = configure_evaluation(cfg)
        arm_root = Path(job['out_dir']) / 'relation'
        state = torch.load(arm_root / 'best.pth', map_location='cpu', weights_only=True)
        model, _ = build_model('relation', cfg['seed'], cfg['paths']['imagenet'], cfg['fusion_options'])
        model.load_state_dict(state['model'], strict=True)
        if model.region_fusion.use_node_geometry:
            raise ValueError('Edge probe requires geometry only on edges, not direct nodes')
        model.cuda().eval()
        dataset = RelationDataset(Path(cfg['paths']['manifests']) / 'pilot_val.csv',
                                  'val', 'relation', cfg['input_size'], cfg['seed'])
        images = []
        with torch.inference_mode():
            for index, row in enumerate(dataset.rows):
                item = dataset[index]
                batch = collate_relation_batch([item])
                inp = {k: v.cuda() for k, v in batch['inputs'].items()}
                def probability(current, auxiliary=False):
                    result = model(**current, return_aux=auxiliary)
                    outputs, debug = result if auxiliary else (result, None)
                    prob = F.interpolate(outputs[-1].float(), size=item['native_hw'],
                                         mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
                    return prob, debug
                full, aux = probability(inp, True)
                saved = np.load(arm_root / ('val_epoch_%02d' % state['epoch']) /
                                (row['image_id'].split('/')[-1] + '.npy'), allow_pickle=False)
                if not np.array_equal(full, saved):
                    raise ValueError('Edge probe full prediction differs from saved probabilities')
                selected = edge_indices(inp, aux['effective_valid'], args.max_edges)
                # GT below is for diagnostics only, never for edge selection/forward.
                labels = candidate_targets(inp['region_masks'], aux['effective_valid'], batch['gt'].cuda())
                with Image.open(row['gt_path']) as im:
                    gt = np.asarray(im.convert('L'), np.float32)
                if gt.max() > 1:
                    gt /= 255.
                full_mae = float(np.abs(full - gt).mean())
                edges = []
                for recipient, sender in selected:
                    current = dict(inp, relation_valid=inp['relation_valid'].clone())
                    current['relation_valid'][0, recipient, sender] = False
                    removed, _ = probability(current)
                    without_mae = float(np.abs(removed - gt).mean())
                    edges.append({'recipient': recipient, 'sender': sender,
                                  'recipient_weak_role': role(inp, recipient), 'sender_weak_role': role(inp, sender),
                                  'recipient_purity': float(labels['purity'][0, recipient]),
                                  'sender_purity': float(labels['purity'][0, sender]),
                                  'mae_removed_minus_full': without_mae - full_mae,
                                  'mean_abs_probability_change': float(np.abs(removed - full).mean())})
                images.append({'image_id': row['image_id'], 'full_mae': full_mae, 'edges': edges})
        deltas = [e['mae_removed_minus_full'] for image in images for e in image['edges']]
        name = 's%d_%s_relation' % (job['seed'], job['condition'])
        report = {'name': name, 'n_images': len(images), 'n_edges': len(deltas),
                  'mean_edge_delta': float(np.mean(deltas)) if deltas else None,
                  'helpful_edges_gt_1e7': sum(d > 1e-7 for d in deltas),
                  'harmful_edges_lt_minus_1e7': sum(d < -1e-7 for d in deltas),
                  'near_zero_edges': sum(abs(d) <= 1e-7 for d in deltas), 'images': images}
        reports.append(report)
        C.write_json(str(root / (name + '_EDGE_UTILITY.json')), report)
        print('RELATION_EDGE_PROBE_ARM_DONE', name, {k: v for k, v in report.items() if k != 'images'}, flush=True)
        del model, state, aux
        gc.collect()
        torch.cuda.empty_cache()
    C.write_json(str(destination), {'status': 'DONE', 'reports': reports,
                                  'evaluation_runtime': runtime_settings,
                                  'max_edges_per_image': args.max_edges,
                                  'selection': 'GT-independent lexicographic valid ordered pair prefix',
                                  'edge_direction': 'sender j -> recipient i; all three heads removed jointly',
                                  'test_evaluated': False, 'used_for_training': False,
                                  'limitation': 'Same-checkpoint functional perturbation with message renormalization; not semantic truth or independent edge samples'})
    print('RELATION_EDGE_UTILITY_DONE', flush=True)


if __name__ == '__main__':
    main()
