"""Synthetic end-to-end report test, no real data/checkpoints/GPU required."""
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.e0e4 import common as C


def main():
    temp_root=ROOT/'.test_tmp';temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cam_report_',dir=temp_root) as directory:
        root=Path(directory);inputs=root/'metric_inputs';(inputs/'RGB').mkdir(parents=True)
        target=np.zeros((16,16),np.float32);target[4:12,4:12]=1
        targets={'TrainVal__a':target,'TrainVal__b':1-target}
        np.savez_compressed(inputs/'targets.npz',**targets)
        index=[{'image_id':'TrainVal/'+k[-1],'metric_key':k} for k in targets]
        C.write_json(str(inputs/'VAL_INDEX.json'),index)
        for key in targets:Image.fromarray(np.zeros((16,16,3),np.uint8)).save(inputs/'RGB'/(key+'.png'))
        records=[]
        for seed in (20261022,20261023):
            for condition,arms in (('raw_zero',('rgb','direct','relation')),('raw_small',('direct','relation')),('scaled_small',('direct','relation'))):
                for arm in arms:
                    tag='s%d_%s_%s'%(seed,condition,arm)
                    probs={k:(v*.7+.1).astype(np.float32) for k,v in targets.items()}
                    mae=float(np.mean([np.abs(probs[k]-targets[k]).mean() for k in targets]))
                    np.savez_compressed(inputs/(tag+'.npz'),**probs)
                    records.append({'seed':seed,'condition':condition,'arm':arm,'tag':tag,'best_epoch':1,'best_mae':mae,'last_mae':mae,
                                    'inference_diagnostics':None,'history':[{'epoch':i,'val_native_mae':mae,'first_batch_fusion_stats':{'relation_relative_l2':1e-3}} for i in (1,2)]})
        C.write_json(str(root/'SWEEP_ANALYSIS.json'),{'records':records,'all_14_runs_done':True,'inference_analysis_completed':True})
        subprocess.run([sys.executable,str(ROOT/'tools/semantic_prior/report_cam_sweep.py'),'--root',str(root)],check=True,stdout=subprocess.DEVNULL)
        report=C.read_json(str(root/'RAW_METRICS_SUMMARY.json'))
        assert len(report['per_run'])==14 and report['n_unique_validation_images']==2
        assert all((root/name).is_file() for name in ('RESULTS.md','LEARNING_CURVES.png','RELATION_RESIDUAL_SCALE.png','CASE_REVIEW.png'))
    print('CAM_SWEEP_REPORT_TESTS_PASSED')


if __name__=='__main__':main()
