"""Per-image checkpoints for the existing tag/segment/package stages."""
import json
import os
from pathlib import Path

from tools.e0e4 import common as C


def write_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    C.write_json(str(temporary), record)
    os.replace(temporary, path)


def existing_record(path, row, config_hash, statuses=None):
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    for key, value in (('image_id', row['image_id']),
                       ('image_sha256', row['image_sha256']),
                       ('config_sha256', config_hash)):
        if record.get(key) != value:
            raise ValueError('Cannot resume changed input/config: ' + str(path))
    if statuses is not None and record.get('status') not in statuses:
        return None  # Retry an explicit tag error, never turn it into empty input.
    return record


def check_candidate_record(record, tag_record, source_root):
    if record.get('tags_source_sha256') != tag_record:
        raise ValueError('Candidate tags changed since checkpoint')
    for tag in record['tags']:
        if tag['status'] not in ('OK', 'EMPTY', 'TEXT_ONLY_ACTION'):
            raise ValueError('Incomplete candidate checkpoint')
        for candidate in tag['candidates']:
            if not (source_root / candidate['mask_path']).is_file():
                raise ValueError('Candidate checkpoint mask is missing')


def existing_package(row, source_root, output_root):
    from lib.region_relations import load_region_package
    destination = output_root / row['image_id']
    metadata = destination / 'regions.json'
    if not metadata.exists():
        return None
    record = json.loads(metadata.read_text(encoding='utf-8'))
    source = source_root / 'candidates' / (row['image_id'] + '.json')
    package = destination / 'regions.npz'
    if (record.get('image_id') != row['image_id'] or
            record.get('image_sha256') != row['image_sha256'] or
            record.get('candidate_source_sha256') != C.sha256_file(str(source)) or
            record.get('package_sha256') != C.sha256_file(str(package)) or
            record.get('candidate_generation_status') not in
            ('HAS_REGIONS', 'NO_CANDIDATES', 'NO_SEGMENTABLE_TAGS')):
        raise ValueError('Cannot resume changed/incomplete package: ' + str(destination))
    arrays = load_region_package(package)
    return dict(image_id=row['image_id'], n_regions=len(arrays['masks']),
                status=record['candidate_generation_status'],
                n_directed_pairs=int(arrays['relation_valid'].sum()),
                roundtrip_equal=True, package=str(package.relative_to(output_root)))
