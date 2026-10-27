"""Verify every train/dev prior, including empty packages; no GT pixel reads."""
import argparse
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.prior_resume import write_record
from lib.region_relations import load_region_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--original', required=True, type=Path)
    args = parser.parse_args()
    summary, all_ids = {}, set()
    for split, expected in (('train', 3652), ('val', 383)):
        rows = C.read_csv(str(args.plan / (split + '.csv')))
        original = C.read_csv(str(args.original / (split + '.csv')))
        if len(rows) != expected or len(original) != expected:
            raise ValueError('Unexpected split size')
        counts, nodes = Counter(), 0
        for row, old in zip(rows, original):
            if ({k:v for k,v in row.items() if k != 'region_package'} !=
                    {k:v for k,v in old.items() if k != 'region_package'}):
                raise ValueError('Sample/GT/split membership changed')
            if row['image_id'] in all_ids:
                raise ValueError('Repeated image ID')
            all_ids.add(row['image_id'])
            path = Path(row['region_package'])
            meta = C.read_json(str(path.with_suffix('.json')))
            if (meta is None or meta['image_id'] != row['image_id'] or
                    meta['image_sha256'] != row['image_sha256'] or
                    meta['package_sha256'] != C.sha256_file(str(path)) or
                    meta.get('gt_used') is not False or meta.get('selection_scores_used') is not False or
                    meta['candidate_generation_status'] not in
                    ('HAS_REGIONS', 'NO_CANDIDATES', 'NO_SEGMENTABLE_TAGS')):
                raise ValueError('Invalid package: ' + str(path))
            arrays = load_region_package(path)
            n = len(arrays['masks'])
            if n != len(meta['nodes']) or list(arrays['masks'].shape[1:]) != meta['mask_hw']:
                raise ValueError('Package/metadata mismatch')
            counts[meta['candidate_generation_status']] += 1
            nodes += n
        summary[split] = dict(n_images=len(rows), n_regions=nodes, status_counts=dict(counts))
    inputs = C.read_csv(str(args.plan / 'PRIOR_INPUT_ALL.csv'))
    if {r['image_id'] for r in inputs} != all_ids:
        raise ValueError('Generation and training scopes differ')
    result = dict(status='ALL_PACKAGES_VERIFIED', n_images=len(all_ids), splits=summary,
                  membership_and_GT_unchanged=True, test_rows_generated=False,
                  empty_priors_retained=True, training_started=False)
    write_record(args.plan / 'FULL_PRIOR_ACCEPTANCE.json', result)
    print('RELATION_FULL_PRIORS_DONE', result, flush=True)


if __name__ == '__main__':
    main()
