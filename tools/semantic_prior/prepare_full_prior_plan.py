"""Reuse prior packages by fixed ID/hash; generate a GT-free pending input list."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C

VALID = {'HAS_REGIONS','NO_CANDIDATES','NO_SEGMENTABLE_TAGS'}
INPUT_KEYS = ('image_id','image_path','image_sha256','width','height')


def reuse(row, candidate):
    for key in ('image_id','image_sha256','image_path','gt_path','split'):
        if row[key] != candidate[key]:
            raise ValueError('Prior source row drift: '+key+' '+row['image_id'])
    path = Path(candidate['region_package'])
    meta = path.with_suffix('.json')
    if not path.is_file() or not meta.is_file():
        return None
    data = json.loads(meta.read_text(encoding='utf-8'))
    if data.get('image_sha256') != row['image_sha256'] or data.get('candidate_generation_status') not in VALID:
        raise ValueError('Existing prior metadata is invalid: '+str(path))
    if data.get('package_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Existing prior package changed: '+str(path))
    return data['candidate_generation_status']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',required=True,type=Path)
    p.add_argument('--extended',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    full = {s:C.read_csv(str(args.manifests/(s+'.csv'))) for s in ('train','val')}
    extra = [r for s in ('train','val') for r in C.read_csv(str(args.extended/('pilot_'+s+'.csv')))]
    extra_by_id = {r['image_id']:r for r in extra}
    all_ids = {r['image_id'] for rows in full.values() for r in rows}
    if len(extra_by_id) != len(extra) or not set(extra_by_id) <= all_ids:
        raise ValueError('Duplicate or out-of-scope prior rows')
    if [len(full[s]) for s in ('train','val')] != [3652,383]:
        raise ValueError('Fixed full data scope changed')
    args.output.mkdir(parents=True,exist_ok=False)
    pending, inputs, reports, new = [],[],{},{}
    for split,rows in full.items():
        statuses, remapped, output_rows = Counter(),0,[]
        for row in rows:
            candidate = extra_by_id.get(row['image_id'],row)
            status = reuse(row,candidate)
            updated = dict(row)
            if status is not None:
                updated['region_package'] = candidate['region_package']
                remapped += updated['region_package'] != row['region_package']
                statuses[status] += 1
            else:
                updated['region_package'] = str(args.output/'priors'/row['image_id']/'regions.npz')
                statuses['PENDING'] += 1
                pending.append({k:row[k] for k in INPUT_KEYS})
            inputs.append({k:row[k] for k in INPUT_KEYS})
            output_rows.append(updated)
        # All sample membership, GT paths and source hashes remain unchanged.
        assert [{k:v for k,v in r.items() if k!='region_package'} for r in rows] == [
            {k:v for k,v in r.items() if k!='region_package'} for r in output_rows]
        new[split] = output_rows
        reports[split] = dict(total=len(rows),reused=len(rows)-statuses['PENDING'],
                             pending=statuses['PENDING'],path_updates=remapped,statuses=dict(statuses))
    for split,rows in new.items():
        C.write_csv(str(args.output/(split+'.csv')),rows)
    C.write_csv(str(args.output/'PRIOR_INPUT_PENDING.csv'),pending,columns=list(INPUT_KEYS))
    C.write_csv(str(args.output/'PRIOR_INPUT_ALL.csv'),inputs,columns=list(INPUT_KEYS))
    C.write_json(str(args.output/'COVERAGE.json'),dict(status='PLAN_READY_NOT_GENERATED',
        splits=reports,n_total=len(inputs),n_reused=len(inputs)-len(pending),n_pending=len(pending),
        source_manifest_sha256={s:C.sha256_file(str(args.manifests/(s+'.csv'))) for s in full},
        source_extended_sha256={s:C.sha256_file(str(args.extended/('pilot_'+s+'.csv'))) for s in full},
        membership_and_GT_unchanged=True,source_packages_rewritten=False,
        test_rows_in_generation_inputs=False,GT_or_split_in_generation_inputs=False,
        valid_empty_priors_retained=True,qwen_or_sam_started=False,training_started=False,
        data_notes='v1 label conflicts remain unchanged; no test-based sample or candidate filtering'))
    print('FULL_PRIOR_PLAN_READY',reports,flush=True)


if __name__=='__main__':
    main()
