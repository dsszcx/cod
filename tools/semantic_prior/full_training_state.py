"""Epoch-boundary checkpoints for the fixed RGB baseline training job."""
import hashlib
import json
import os
from pathlib import Path

import torch


def signature(config, manifest_hashes):
    payload = {'config': {k: v for k, v in config.items() if not k.startswith('_')},
               'manifest_hashes': manifest_hashes}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config['lr_milestones'], gamma=config['lr_gamma'])


def save_checkpoint(path, model, optimizer, scheduler, **metadata):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), **metadata), temp)
    os.replace(temp, path)


def latest_checkpoint(root):
    """Best can be one completed epoch newer if interruption preceded last save."""
    chosen = None
    for name in ('last.pth', 'best.pth'):
        path = Path(root)/name
        if path.is_file():
            state = torch.load(path, map_location='cpu', weights_only=True)
            if chosen is None or state['epoch'] > chosen[1]['epoch']:
                chosen = (path, state)
    if chosen is None:
        raise FileNotFoundError('No completed-epoch checkpoint to resume')
    return chosen


def restore(state, model, optimizer, scheduler, expected_signature):
    if state['run_signature'] != expected_signature:
        raise ValueError('Resume config or split differs from the original run')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
