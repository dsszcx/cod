"""Scientific validation comparison: native RGB/GT and fixed-scale probabilities."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.e0e4 import common as C
from tools.semantic_prior.common import resolve


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/relation_pilot10_20261026.yaml')
    args=ap.parse_args()
    cfg=C.load_config(str(resolve(args.config)))
    root=resolve(cfg['paths']['out_dir'])
    report_path=root/'ANALYSIS_V2.json'
    if not report_path.exists(): report_path=root/'ANALYSIS.json'
    report=C.read_json(str(report_path))
    summary=C.read_json(str(root/'SUMMARY.json'))
    rows={r['image_id']:r for r in C.read_csv(str(resolve(cfg['paths']['manifests'])/'pilot_val.csv'))}
    cases=report['per_image_deltas'][-3:][::-1]
    fig,axes=plt.subplots(len(cases),5,figsize=(15,9),constrained_layout=True)
    for i,case in enumerate(cases):
        row=rows[case['image_id']]
        with Image.open(row['image_path']) as im: rgb=np.asarray(im.convert('RGB'))
        with Image.open(row['gt_path']) as im: gt=np.asarray(im.convert('L'),np.float32)
        if gt.max()>1: gt/=255
        images=[rgb,gt]
        for arm in cfg['arms']:
            epoch=next(a['best_epoch'] for a in summary['arms'] if a['arm']==arm)
            images.append(np.load(root/arm/('val_epoch_%02d'%epoch)/(case['image_id'].split('/')[-1]+'.npy')))
        for j,image in enumerate(images):
            axes[i,j].imshow(image) if j==0 else axes[i,j].imshow(image,cmap='gray',vmin=0,vmax=1)
            title=['Original','GT','RGB','Direct','Relation'][j]
            if j>=2: title+=' MAE %.4f'%case[['rgb_mae','direct_mae','relation_mae'][j-2]]
            axes[i,j].set_title(title,fontsize=11)
            axes[i,j].axis('off')
        axes[i,0].set_xlabel(case['image_id'].split('/')[-1],fontsize=7)
    fig.suptitle('Worst 3 relation-minus-direct cases on the SAME %d-image pilot validation; probabilities fixed [0,1]'%summary['n_val'],fontsize=12)
    fig.savefig(root/'VALIDATION_WORST3.png',dpi=140)
    plt.close(fig)
    print('VALIDATION_REVIEW_DONE',flush=True)


if __name__=='__main__':main()
