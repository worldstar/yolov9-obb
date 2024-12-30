import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr
from utils.rboxs_utils import pi, poly2rbox, regular_theta

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    Args:
        Dataset.labels (list): n_imgs * array(num_gt_perimg, [cls_id, poly])
        Dataset.shapes (array): (n_imgs, [ori_img_width, ori_img_height])
    Returns:
        
    """
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    #shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    #scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    #wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh
    min_ratios = imgsz  / dataset.shapes.max(1, keepdims=True) # 
    scales = np.random.uniform(0.9, 1.1, size=(min_ratios.shape[0], 1))  # augment scale

    ls_edges = []
    for ratio, labels in zip(min_ratios * scales, dataset.labels): # labels (array): (num_gt_perimg, [cls_id, poly])
        rboxes = poly2rbox(labels[:, 1:] * ratio)
        if len(rboxes):
            ls_edges.append(rboxes[:, 2:4])
    ls_edges = torch.tensor(np.concatenate(ls_edges)).float()
    ls_edges = ls_edges[(ls_edges >= 5.0).any(1)]  # filter > 5 pixels, anchor 宽高不能都小于5

    def metric(k):  # compute metric
        #r = wh[:, None] / k[None]
        r = ls_edges[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    #stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    #anchors = m.anchors.clone() * stride  # current anchors
    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    print("Anchors in train.py:", anchors)
    
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = m.anchors.numel() // 2  # number of anchors
        #anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            #m.anchors[:] = anchors.clone().view_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss, featuremap stride pixel
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            #m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        #_, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        _, best = metric(torch.tensor(k, dtype=torch.float32), ls_edges)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        #x, best = metric(k, wh0)
        x, best = metric(k, ls_edges0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    """
    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh
    """
    # Get label l s
    min_ratios = img_size  / dataset.shapes.max(1, keepdims=True) # 
    ls_edges0 = []
    for ratio, labels in zip(min_ratios, dataset.labels): # labels (array): (num_gt_perimg, [cls_id, poly])
        rboxes = poly2rbox(labels[:, 1:] * ratio)
        if len(rboxes):
            ls_edges0.append(rboxes[:, 2:4])
    ls_edges0 = np.concatenate(ls_edges0)

    # Filter
    """
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1
    """
    i = (ls_edges0 < 5.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found. {i} of {len(ls_edges0)} poly labels are < 5 pixels in size.')
    ls_edges = ls_edges0[(ls_edges0 >= 5.0).any(1)]  # filter > 5 pixels

    # Kmeans init/calculation
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(ls_edges)} points...')
        #assert n <= len(wh)  # apply overdetermined constraint
        #s = wh.std(0)  # sigmas for whitening
        #k = kmeans(wh / s, n, iter=30)[0] * s  # points
        s = ls_edges.std(0)  # sigmas for whitening
        k, dist = kmeans(ls_edges / s, n, iter=30)  # points, mean distance
        #assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
        assert len(k) == n, f'{PREFIX}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    #wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    #k = print_results(k, verbose=False)
    k *= s
    ls_edges = torch.tensor(ls_edges, dtype=torch.float32)  # filtered
    ls_edges0 = torch.tensor(ls_edges0, dtype=torch.float32)  # unfiltered
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
