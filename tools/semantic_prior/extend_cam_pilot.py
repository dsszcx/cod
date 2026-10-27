"""Fixed CAM-only larger pilot, reusing valid existing packages without GT selection."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from tools.semantic_prior.prepare_training import pilot_rows
from tools.semantic_prior.prepare_cam_only import category, assert_splits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/relation_cam_extended_20261026.yaml')
    cfg=C.load_config(str(resolve(ap.parse_args().config)))
    root=resolve(cfg['paths']['out_dir']); source=resolve(cfg['paths']['source_manifests'])
    old={s:C.read_csv(str(source/(s+'.csv'))) for s in ('train','val','test')}
    assert_splits(old['train'],old['val'],old['test'])
    selected={s:pilot_rows(old[s],cfg['pilot'][s+'_images']) for s in ('train','val')}
    missing=[]; reused=0
    for rows in selected.values():
        for row in rows:
            if category(row)=='NonCAM': raise ValueError('NonCAM')
            path=Path(row['region_package'])
            if path.is_file():
                meta=C.read_json(str(path.with_suffix('.json')))
                if meta is None or meta['image_sha256']!=row['image_sha256'] or C.sha256_file(str(path))!=meta['package_sha256']:
                    raise ValueError('Invalid cached package')
                reused+=1
            else:
                row['region_package']=str(root/'priors'/row['image_id']/'regions.npz')
                missing.append(row)
    root.mkdir(parents=True,exist_ok=False)
    by_id={r['image_id']:r for rows in selected.values() for r in rows}
    for s in ('train','val'):
        C.write_csv(str(root/('pilot_'+s+'.csv')),selected[s])
        C.write_csv(str(root/(s+'.csv')),[by_id.get(r['image_id'],r) for r in old[s]])
    keys=('image_id','image_path','image_sha256','width','height')
    C.write_csv(str(root/'MISSING_PRIOR_INPUT.csv'),[{k:r[k] for k in keys} for r in missing])
    C.write_json(str(root/'EXTENSION_REPORT.json'),{'n_train':len(selected['train']),'n_val':len(selected['val']),
        'existing_packages_reused':reused,'missing_packages':len(missing),'selection':'prefix of frozen hash-group order; no GT/prediction selection',
        'split_membership_preserved':True,'test_evaluated':False,'test_manifest_source':str(source/'test.csv'),
        'config_sha256':cfg['_sha256']})
    print('CAM_PILOT_EXTENDED',len(selected['train']),len(selected['val']),'reused',reused,'missing',len(missing),flush=True)


if __name__=='__main__':main()
