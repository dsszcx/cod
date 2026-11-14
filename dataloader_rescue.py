"""Provenance-preserving old/rescue pairing with identical augmentation.

GT is returned once outside both input dictionaries. Candidate membership is
derived only from stored candidate IDs, never from GT or COD predictions.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader_relation import RelationDataset, collate_relation_batch
from lib.region_relations import collate_region_samples


class PairedRescueDataset(Dataset):
    def __init__(self, original_manifest, rescue_manifest, split, size=352, seed=20261023):
        self.old = RelationDataset(original_manifest, split, 'direct', size, seed)
        self.new = RelationDataset(rescue_manifest, split, 'direct', size, seed)
        if len(self.old) != len(self.new):
            raise ValueError('Manifest sizes differ')
        self.rows = self.old.rows
        self.indices = []
        for a, b in zip(self.old.rows, self.new.rows):
            if {k: v for k, v in a.items() if k != 'region_package'} != {k: v for k, v in b.items() if k != 'region_package'}:
                raise ValueError('Image/GT/split membership or order changed')
            am = json.loads(Path(a['region_package']).with_suffix('.json').read_text())
            bm = json.loads(Path(b['region_package']).with_suffix('.json').read_text())
            an = sorted(am['nodes'], key=lambda n: n['mask_index'])
            bn = sorted(bm['nodes'], key=lambda n: n['mask_index'])
            if [n['mask_index'] for n in an] != list(range(len(an))) or [n['mask_index'] for n in bn] != list(range(len(bn))):
                raise ValueError('Candidate indices are not unique and contiguous')
            old_ids = {n['candidate_id'] for n in an}
            lookup = {n['candidate_id']: n['mask_index'] for n in bn}
            if len(old_ids) != len(an) or len(lookup) != len(bn) or not old_ids <= lookup.keys():
                raise ValueError('Duplicate or missing original candidate IDs')
            self.indices.append(([lookup[n['candidate_id']] for n in an],
                                 [n['mask_index'] for n in bn if n['candidate_id'] not in old_ids]))

    def __len__(self):
        return len(self.old)

    def set_epoch(self, epoch):
        self.old.set_epoch(epoch)
        self.new.set_epoch(epoch)

    def __getitem__(self, index):
        old, new = self.old[index], self.new[index]
        old_indices, rescue_indices = self.indices[index]
        if (old['spatial_plan'] != new['spatial_plan'] or not torch.equal(old['gt'], new['gt'])
                or not np.array_equal(old['inputs']['rgb'], new['inputs']['rgb'])):
            raise ValueError('Paired augmentation differs')
        for key in ('masks', 'region_valid', 'role_hint', 'role_hint_valid'):
            if not np.array_equal(old['inputs'][key], new['inputs'][key][old_indices]):
                raise ValueError('Original candidate content/order changed: ' + key)
        rescue = {key: new['inputs'][key][rescue_indices] for key in ('masks', 'region_valid', 'role_hint', 'role_hint_valid')}
        rescue['rgb'] = new['inputs']['rgb']
        return dict(original=old, rescue=rescue, image_id=old['image_id'], n_added=len(rescue_indices))


def collate_rescue_batch(items):
    original = collate_relation_batch([x['original'] for x in items])
    rescue = collate_region_samples([x['rescue'] for x in items])
    return dict(original_inputs=original['inputs'],
                rescue_inputs={k: rescue[k] for k in ('region_masks', 'region_valid', 'role_hint', 'role_hint_valid')},
                gt=original['gt'], image_ids=original['image_ids'], native_hw=original['native_hw'],
                n_added=[x['n_added'] for x in items])
