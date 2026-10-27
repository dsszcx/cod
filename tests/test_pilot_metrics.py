from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.pilot_metrics import raw_metrics, PRIMARY_METRICS, require_primary_metrics


def main():
    gt=np.zeros((16,16),np.float32);gt[4:12,4:12]=1
    pred=np.linspace(.2,.3,256,dtype=np.float32).reshape(16,16)
    scores=raw_metrics(pred,gt)
    assert PRIMARY_METRICS == ('mae','sm','mean_e','wfm')
    require_primary_metrics(scores)
    for invalid in ({'mae': .1}, {**scores, 'wfm': float('nan')}):
        try:
            require_primary_metrics(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete/nonfinite primary metrics accepted')
    assert abs(scores['mae']-np.abs(pred-gt).mean())<1e-8
    stretched=(pred-pred.min())/(pred.max()-pred.min())
    assert abs(scores['mae']-raw_metrics(stretched,gt)['mae'])>.01
    assert scores['foreground_fraction']==0 and scores['dice_05']==0
    assert all(np.isfinite(scores[k]) for k in ('mae','sm','wfm','mean_e','iou_05','dice_05'))
    assert raw_metrics(np.zeros_like(gt),np.zeros_like(gt))['iou_05']==1
    print('RAW_PILOT_METRICS_TESTS_PASSED')


if __name__=='__main__':main()
