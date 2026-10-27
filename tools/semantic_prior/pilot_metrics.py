"""Raw probability metrics; explicitly bypass PySOD's default per-image minmax."""
import inspect
import numpy as np

PRIMARY_METRICS = ('mae', 'sm', 'mean_e', 'wfm')


def require_primary_metrics(scores):
    """A completed experiment must report all four primary COD metrics."""
    missing = [key for key in PRIMARY_METRICS if key not in scores]
    if missing:
        raise ValueError('Incomplete primary COD metrics: ' + ', '.join(missing))
    if any(not np.isfinite(float(scores[key])) for key in PRIMARY_METRICS):
        raise ValueError('Primary COD metrics must all be finite')


def raw_metrics(prob, target):
    import py_sod_metrics as M
    prob=np.asarray(prob,np.float32);target=np.asarray(target,np.float32)
    if prob.shape!=target.shape or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:
        raise ValueError('Invalid raw probability/GT shapes or range')
    gt=target>=.5
    result={'mae':float(np.abs(prob-target).mean())}
    for name,cls in (('sm',M.Smeasure),('wfm',M.WeightedFmeasure),('em',M.Emeasure)):
        metric=cls()
        if 'normalize' not in inspect.signature(metric.step).parameters:
            raise RuntimeError('PySOD backend lacks explicit normalize=False; do not silently minmax')
        metric.step(prob,gt,normalize=False)
        values=metric.get_results()[name]
        if name=='em':result['mean_e']=float(np.mean(values['curve']))
        else:result[name]=float(values)
    binary=prob>=.5
    inter=int((binary&gt).sum());union=int((binary|gt).sum())
    result['iou_05']=inter/union if union else 1.
    denom=int(binary.sum()+gt.sum())
    result['dice_05']=2*inter/denom if denom else 1.
    result['foreground_fraction']=float(binary.mean())
    result['gt_foreground_fraction']=float(gt.mean())
    result['gt_nonbinary_pixels']=int(((target!=0)&(target!=1)).sum())
    require_primary_metrics(result)
    return result
