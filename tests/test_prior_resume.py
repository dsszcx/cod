"""Resume completed images, preserve empties and reject stale inputs."""
from pathlib import Path
import contextlib
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.e0e4 import common as C
from tools.semantic_prior.prior_resume import existing_record, write_record, check_candidate_record
from tools.semantic_prior.pack_regions import pack_one
from tools.semantic_prior import generate_manifest_tags as tags
from tools.semantic_prior import segment_tags as segmentation


def must_fail(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Accepted stale checkpoint')


def main():
    (ROOT / '.test_tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / '.test_tmp') as temporary:
        root = Path(temporary)
        image = root / 'image.png'
        Image.fromarray(np.ones((9, 11, 3), np.uint8) * 128).save(image)
        row = dict(image_id='TrainVal/fixture', image_path=str(image),
                   image_sha256=C.sha256_file(str(image)), width='11', height='9')
        checkpoint = root / 'tag.json'
        rec = dict(image_id=row['image_id'], image_sha256=row['image_sha256'],
                   config_sha256='fixed', status='EMPTY', tags=[])
        write_record(checkpoint, rec)
        assert existing_record(checkpoint, row, 'fixed', ('OK', 'EMPTY')) == rec
        assert not checkpoint.with_name('tag.json.tmp').exists()
        must_fail(lambda: existing_record(checkpoint, row, 'changed'))
        rec['status'] = 'ERROR'
        rec.update(raw_replies=['bad schema'], attempts=[dict(raw='bad schema', error='invalid')])
        write_record(checkpoint, rec)
        assert existing_record(checkpoint, row, 'fixed', ('OK', 'EMPTY')) is None
        from unittest.mock import Mock
        model = Mock()
        model.caption.return_value = '{"tags":[{"tag":"surface","kind":"context"}]}'
        repaired = tags.repair_record(model, row, rec, dict(max_tags=8, max_words=6))
        assert repaired['status'] == 'OK' and len(repaired['attempts']) == 2
        assert rec['raw_replies'] == ['bad schema']  # Original is unchanged.
        assert model.caption.call_count == 1
        untyped, _, recovery = tags.parse_repair_tags('{"tags":["public bench", "sidewalk"]}', dict(max_tags=8, max_words=6))
        assert len(untyped) == 2 and all(t['kind'] == 'unknown' for t in untyped)
        assert recovery == 'untyped_list_unknown_roles_v1'
        model.caption.return_value = '{not valid JSON'
        assert tags.repair_record(model, row, rec, dict(max_tags=8, max_words=6))['status'] == 'ERROR'
        check_candidate_record(dict(tags_source_sha256='taghash', tags=[]), 'taghash', root)
        must_fail(lambda: check_candidate_record(dict(tags_source_sha256='old'), 'new', root))

        # A valid empty package is kept. Partial package directories can recover.
        source = root / 'source'
        candidates = dict(image_id=row['image_id'], image_sha256=row['image_sha256'],
                          source_size=[9, 11], tags=[])
        candidate_path = source / 'candidates' / (row['image_id'] + '.json')
        write_record(candidate_path, candidates)
        output = root / 'packages'
        (output / row['image_id']).mkdir(parents=True)  # Simulated interrupted pack.
        first = pack_one(row, source, output, resume=True)
        assert first['status'] == 'NO_SEGMENTABLE_TAGS' and first['n_regions'] == 0
        package = output / row['image_id'] / 'regions.npz'
        before = package.stat().st_mtime_ns
        assert pack_one(row, source, output, resume=True) == first
        assert package.stat().st_mtime_ns == before
        candidates['source_size'] = [10, 11]
        write_record(candidate_path, candidates)
        must_fail(lambda: pack_one(row, source, output, resume=True))

        # CLI resume reuses tags without loading Qwen; config changes fail.
        inputs, config, tag_out = root / 'inputs.csv', root / 'config.yaml', root / 'tags_out'
        C.write_csv(str(inputs), [row])
        cfg = dict(_path=str(config), _sha256='x', expected_images=1,
                   paths=dict(source_manifest=str(inputs), out_dir=str(tag_out), qwen_model='unused'),
                   tags=dict(max_tags=8, max_words=6, max_tokens=700, retry_invalid=1,
                             prompt_version='concrete_components_no_example_v2'))
        config.write_text('fixture', encoding='utf-8')
        cfg['_sha256'] = C.sha256_file(str(config))
        rec.update(status='EMPTY', config_sha256=cfg['_sha256'])
        write_record(tag_out / 'tags' / (row['image_id'] + '.json'), rec)
        with patch.object(sys, 'argv', ['tags', '--config', str(config), '--resume']), \
             patch.object(C, 'load_config', return_value=cfg), \
             patch.object(tags, 'QwenCaptioner', side_effect=AssertionError('Qwen should not load')):
            assert tags.main() == 0
        assert C.read_json(str(tag_out / 'TAG_SUMMARY.json'))['complete'] is True

        # Mock GPU kernels, not file I/O: resuming SAM3 preserves table counts
        # and must not rebuild the model or execute any prompt again.
        import torch
        rec['tags'] = [dict(tag_id='t01', tag='surface', kind='unknown', segmentable=True),
                       dict(tag_id='t02', tag='resting', kind='action', segmentable=False)]
        write_record(tag_out / 'tags' / (row['image_id'] + '.json'), rec)
        cfg.update(sam3=dict(confidence_threshold=.5, max_masks_per_tag=2))
        cfg['paths']['sam3_ckpt'] = 'unused'

        class Processor:
            calls = 0

            def set_image(self, image):
                self.calls += 1
                return {}

            def set_text_prompt(self, prompt, state):
                return dict(scores=torch.tensor([.8]), masks=torch.ones((1, 1, 9, 11)))

        processor = Processor()
        with patch.object(sys, 'argv', ['segment', '--config', str(config), '--resume']), \
             patch.object(C, 'load_config', return_value=cfg), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()), \
             patch.object(segmentation, 'load_processor', return_value=processor) as loader:
            segmentation.main()
            first_table = C.read_csv(str(tag_out / 'CANDIDATE_TABLE.csv'))
            segmentation.main()
            assert loader.call_count == 1 and processor.calls == 1
        assert C.read_csv(str(tag_out / 'CANDIDATE_TABLE.csv')) == first_table
        candidate_summary = C.read_json(str(tag_out / 'CANDIDATE_SUMMARY.json'))
        assert candidate_summary['n_candidates'] == 1
        assert candidate_summary['tag_status_counts'] == dict(OK=1, TEXT_ONLY_ACTION=1)
        nonempty = pack_one(row, tag_out, root / 'nonempty', resume=True)
        assert nonempty['n_regions'] == 1 and nonempty['status'] == 'HAS_REGIONS'
        from lib.region_relations import load_region_package
        arrays = load_region_package(root / 'nonempty' / row['image_id'] / 'regions.npz')
        assert not arrays['role_hint_valid'].any() and arrays['role_hint'][0, 3] == 1
    print('PRIOR_RESUME_TESTS_PASSED', flush=True)


if __name__ == '__main__':
    main()
