"""Python protocol check against upstream SINet-V2 eval/*.m, not a MATLAB run.

S/weighted F formulas follow the upstream implementation (Apache-2.0).
SciPy's nearest-foreground tie breaking can differ from MATLAB bwdist.
"""
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

EPS = np.finfo(np.float64).eps
THRESHOLDS = np.arange(255, -1, -1, dtype=np.float64) / 255


def prediction_to_png(prob):
    """Official sigmoid->minmax->legacy SciPy bytescale uint8 adapter."""
    p = np.asarray(prob)
    if p.ndim != 2 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Expected a finite HxW sigmoid probability map')
    p = (p-p.min())/(p.max()-p.min()+1e-8)
    span = p.max()-p.min()
    return (((p-p.min())*(255/(span if span else 1))).clip(0, 255)+.5).astype(np.uint8)


def mapminmax(pred_u8):
    p = np.asarray(pred_u8, np.float64) / 255
    lo, hi = p.min(), p.max()
    # MATLAB mapminmax leaves constant rows unchanged, not zeroed.
    return (p - lo) / (hi - lo) if hi > lo else p


def ssim(p, g):
    if not p.size:
        return float('nan')  # Preserve undefined upstream empty-region behavior.
    x, y = p.mean(), g.mean()
    denom = p.size - 1 + EPS
    vx, vy = ((p-x)**2).sum()/denom, ((g-y)**2).sum()/denom
    cov = ((p-x)*(g-y)).sum()/denom
    a, b = 4*x*y*cov, (x*x+y*y)*(vx+vy)
    return a/(b+EPS) if a != 0 else float(b == 0)


def structure(p, g):
    y = g.mean()
    if y == 0:
        return 1-p.mean()
    if y == 1:
        return p.mean()

    def obj(v):
        std = v.std(ddof=1) if v.size > 1 else 0.
        return 2*v.mean()/(v.mean()**2+1+std+EPS)

    object_score = y*obj(p[g])+(1-y)*obj(1-p[~g])
    # MATLAB round is half-away-from-zero, with one-based coordinates.
    cy, cx = np.floor(np.argwhere(g).mean(axis=0)+1+0.5).astype(int)
    h, w = g.shape
    weights = [cx*cy/(h*w), (w-cx)*cy/(h*w), cx*(h-cy)/(h*w)]
    weights.append(1-sum(weights))
    regions = [(slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, w)),
               (slice(cy, h), slice(0, cx)), (slice(cy, h), slice(cx, w))]
    region_score = sum(weight*ssim(p[r], g[r]) for weight, r in zip(weights, regions))
    return max(0., .5*(object_score+region_score)) if np.isfinite(region_score) else float('nan')


def weighted_f(p, g):
    if not g.any():
        raise ValueError('Empty GT is outside the CAM-only benchmark protocol')
    dist, indices = distance_transform_edt(~g, return_indices=True)
    error = np.abs(p-g)
    extended = error.copy()
    extended[~g] = error[indices[0][~g], indices[1][~g]]
    yy, xx = np.ogrid[-3:4, -3:4]
    kernel = np.exp(-(xx*xx+yy*yy)/50.)
    kernel /= kernel.sum()
    smooth = convolve(extended, kernel, mode='constant', cval=0)
    minimum = np.where(g & (smooth < error), smooth, error)
    weights = np.where(g, 1., 2-np.exp(np.log(.5)/5*dist))
    ew = minimum*weights
    tp, fp = g.sum()-ew[g].sum(), ew[~g].sum()
    recall, precision = 1-ew[g].mean(), tp/(EPS+tp+fp)
    return 2*recall*precision/(EPS+recall+precision)


def e_curve(p, g):
    """Exact strict p > threshold; no second uint8 histogram quantization."""
    fg = np.sort(p[g])
    bg = np.sort(p[~g])
    tp = fg.size-np.searchsorted(fg, THRESHOLDS, side='right')
    fp = bg.size-np.searchsorted(bg, THRESHOLDS, side='right')
    size, truth = p.size, g.sum()
    predicted = tp+fp
    if truth == 0:
        return (size-predicted)/(size-1+EPS)
    if truth == size:
        return predicted/(size-1+EPS)
    mg, mp = truth/size, predicted/size
    total = np.zeros(256)
    for count, pv, gv in [(tp, 1-mp, 1-mg), (fp, 1-mp, -mg),
                          (truth-tp, -mp, 1-mg), (size-truth-fp, -mp, -mg)]:
        align = 2*pv*gv/(pv*pv+gv*gv+EPS)
        total += count*(align+1)**2/4
    return total/(size-1+EPS)


def evaluate(pred_u8, gt_u8):
    if pred_u8.shape != gt_u8.shape or pred_u8.ndim != 2:
        raise ValueError('Prediction/GT size mismatch; never resize benchmark GT')
    if pred_u8.dtype != np.uint8 or gt_u8.dtype != np.uint8:
        raise TypeError('Benchmark evaluator expects grayscale uint8 PNG arrays')
    p, g = mapminmax(pred_u8), gt_u8 > 128
    if not g.any():
        raise ValueError('Empty GT is outside the CAM-only benchmark protocol')
    result = dict(mae=float(np.abs(p-g).mean()), sm=float(structure(p, g)),
                  mean_e=float(e_curve(p, g).mean()), wfm=float(weighted_f(p, g)))
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite metric: retain sample and diagnose, do not silently skip')
    return result
