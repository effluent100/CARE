from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
import torch
try:
    from skimage import measure
except ModuleNotFoundError:
    measure = None
    from scipy import ndimage


def _iter_mask_regions(mask):
    if measure is not None:
        for region in measure.regionprops(measure.label(mask)):
            yield region.coords, region.area
        return

    labeled, num_regions = ndimage.label(mask)
    if num_regions == 0:
        return
    objects = ndimage.find_objects(labeled)
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        local = labeled[slices] == label_id
        coords = np.argwhere(local)
        offsets = np.array([sl.start for sl in slices])
        coords = coords + offsets
        yield coords, int(local.sum())


# ---------------------------------------------------------------------------
# GPU-exact pixel AUROC / AUPRC.
# These reproduce sklearn.roc_auc_score / average_precision_score to within
# float32 rounding (|diff| < 1e-7) but run in ~0.03s instead of ~10s per object
# by doing the O(N log N) sort on the (otherwise idle) GPU. Fall back to sklearn
# on CPU when CUDA is unavailable.
# ---------------------------------------------------------------------------
def _gpu_auroc(gt, pr, device):
    """Exact ROC AUC via the Mann-Whitney U statistic with average ranks."""
    y = torch.as_tensor(np.asarray(gt).ravel(), device=device).float()
    s = torch.as_tensor(np.asarray(pr).ravel(), device=device).float()
    n_pos = y.sum()
    n_neg = y.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    n = s.numel()
    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float64)
    diff = torch.ones(n, dtype=torch.bool, device=device)
    diff[1:] = s_sorted[1:] != s_sorted[:-1]
    group_id = torch.cumsum(diff.long(), 0) - 1
    ng = int(group_id[-1].item()) + 1
    sum_ranks = torch.zeros(ng, dtype=torch.float64, device=device)
    cnt = torch.zeros(ng, dtype=torch.float64, device=device)
    sum_ranks.scatter_add_(0, group_id, ranks)
    cnt.scatter_add_(0, group_id, torch.ones(n, dtype=torch.float64, device=device))
    avg_ranks = (sum_ranks / cnt)[group_id]
    rank_pos = avg_ranks[y_sorted.bool()].sum()
    auroc = (rank_pos - n_pos.double() * (n_pos.double() + 1) / 2) / (n_pos.double() * n_neg.double())
    return float(auroc.item())


def _gpu_auprc(gt, pr, device):
    """Exact average precision (matches sklearn.average_precision_score)."""
    y = torch.as_tensor(np.asarray(gt).ravel(), device=device).float()
    s = torch.as_tensor(np.asarray(pr).ravel(), device=device).float()
    if y.sum() == 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    y_sorted = y[order]
    s_sorted = s[order]
    distinct = torch.ones(s_sorted.numel(), dtype=torch.bool, device=device)
    distinct[:-1] = s_sorted[1:] != s_sorted[:-1]
    tps = torch.cumsum(y_sorted, 0)
    fps = torch.cumsum(1 - y_sorted, 0)
    idx = torch.nonzero(distinct, as_tuple=False).squeeze(1)
    tp = tps[idx]
    fp = fps[idx]
    precision = tp / (tp + fp)
    recall = tp / tps[-1]
    recall_prev = torch.cat([torch.zeros(1, device=device, dtype=recall.dtype), recall[:-1]])
    ap = torch.sum((recall - recall_prev) * precision)
    return float(ap.item())


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    #
    # Numerically identical to the original threshold sweep, but the connected
    # component labeling (measure.label / regionprops) and the background pixel
    # selection are threshold-independent, so they are computed ONCE up front
    # instead of once per threshold. The per-region true-positive counting is
    # then vectorized with np.bincount. This removes the dominant CPU cost.
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step

    # Precompute, once, the anomaly-map values under each GT region and the
    # region areas (both threshold-independent).
    region_values = []   # amap value for every pixel belonging to a GT region
    region_areas = []
    for amap, mask in zip(amaps, masks):
        for coords, area in _iter_mask_regions(mask):
            region_values.append(amap[coords[:, 0], coords[:, 1]])
            region_areas.append(area)
    region_areas = np.array(region_areas, dtype=np.float64)
    n_regions = len(region_values)

    if n_regions:
        all_vals = np.concatenate(region_values)
        region_ids = np.concatenate(
            [np.full(len(v), i, dtype=np.int64) for i, v in enumerate(region_values)]
        )
    else:
        all_vals = np.empty(0, dtype=amaps.dtype)
        region_ids = np.empty(0, dtype=np.int64)

    # Precompute the background (inverse-mask) anomaly values once.
    inverse_masks = 1 - masks
    bg_bool = inverse_masks.astype(bool)
    bg_values = amaps[bg_bool]
    bg_total = inverse_masks.sum()

    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        hits = all_vals > th
        tp = np.bincount(region_ids[hits], minlength=n_regions).astype(np.float64)
        pros.append((tp / region_areas).mean())
        fprs.append((bg_values > th).sum() / bg_total)

    pros, fprs = np.array(pros), np.array(fprs)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc



def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if len(np.unique(gt)) < 2:
        return float("nan")
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    if metric == 'pixel-auroc':
        performance = _gpu_auroc(gt, pr, device) if use_gpu else roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    elif metric == 'pixel-auprc':
        performance = _gpu_auprc(gt, pr, device) if use_gpu else average_precision_score(gt.ravel(), pr.ravel())
    return performance

    