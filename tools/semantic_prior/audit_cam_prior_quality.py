"""Post-generation GT-based quality audit. Never changes sample/prompt selection."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve
from lib.region_relations import load_region_package


def main():
    root=resolve('outputs/RELATION_CAM_EXTENDED_20261026_V1')
    target=root/'PRIOR_QUALITY_AUDIT.json'
    if target.exists():raise FileExistsError(str(target))
    records=[]
    for split in ('train','val'):
        for row in C.read_csv(str(root/('pilot_'+split+'.csv'))):
            sample=load_region_package(row['region_package'])
            masks=sample['masks'].astype(bool)
            with Image.open(row['gt_path']) as im:gt=np.asarray(im.convert('L'),np.float32)
            if gt.max()>1:gt/=255.
            fg=gt>=.5;area=int(fg.sum())
            union=masks.any(0) if len(masks) else np.zeros_like(fg)
            intersections=(masks&fg).sum((1,2))
            unions=(masks|fg).sum((1,2))
            ious=np.divide(intersections,unions,out=np.zeros(len(masks),np.float64),where=unions>0)
            records.append({'image_id':row['image_id'],'split':split,'n_regions':len(masks),
                'gt_empty':not bool(area),'zero_prediction_mae':float(gt.mean()),
                'best_candidate_iou_oracle':float(ious.max()) if len(ious) else 0.,
                'union_foreground_recall':float((union&fg).sum()/area) if area else None,
                'union_precision':float((union&fg).sum()/union.sum()) if union.any() else 0.,
                'n_whole':int((sample['role_hint'][:,0]>.5).sum()),'n_part':int((sample['role_hint'][:,1]>.5).sum()),
                'n_context':int((sample['role_hint'][:,2]>.5).sum())})
    summaries={}
    for split in ('train','val'):
        rows=[r for r in records if r['split']==split]
        summaries[split]={'n_images':len(rows),'n_regions':sum(r['n_regions'] for r in rows),
            'empty_prior_images':sum(r['n_regions']==0 for r in rows),'empty_gt_images':sum(r['gt_empty'] for r in rows),
            'mean_zero_prediction_mae':float(np.mean([r['zero_prediction_mae'] for r in rows])),
            'mean_best_candidate_iou_oracle':float(np.mean([r['best_candidate_iou_oracle'] for r in rows])),
            'images_with_candidate_iou_ge_05':sum(r['best_candidate_iou_oracle']>=.5 for r in rows),
            'mean_union_recall':float(np.mean([r['union_foreground_recall'] for r in rows if r['union_foreground_recall'] is not None])),
            'mean_union_precision':float(np.mean([r['union_precision'] for r in rows]))}
    C.write_json(str(target),{'splits':summaries,'per_image':records,
        'audit_after_generation':True,'gt_used_for_audit_only':True,'samples_removed':False,'prompts_changed':False,
        'oracle_warning':'Best candidate IoU uses GT and is an optimistic diagnostic, never a selectable prediction',
        'test_evaluated':False})
    print('CAM_PRIOR_QUALITY_AUDIT',summaries,flush=True)


if __name__=='__main__':main()
