import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel, is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

"""
class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        bs = p[0].shape[0]  # batch size
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses
        tcls, tbox, indices = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # tgt obj

            n_labels = b.shape[0]  # number of labels
            if n_labels:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                # pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                # pwh = (0.0 + (pwh - 1.09861).sigmoid() * 4) * anchors[i]
                # pwh = (0.33333 + (pwh - 1.09861).sigmoid() * 2.66667) * anchors[i]
                # pwh = (0.25 + (pwh - 1.38629).sigmoid() * 3.75) * anchors[i]
                # pwh = (0.20 + (pwh - 1.60944).sigmoid() * 4.8) * anchors[i]
                # pwh = (0.16667 + (pwh - 1.79175).sigmoid() * 5.83333) * anchors[i]
                pxy = pxy.sigmoid() * 1.6 - 0.3
                pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                loss[0] += (1.0 - iou).mean()  # box loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, gj, gi, iou = b[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n_labels), tcls[i]] = self.cp
                    loss[2] += self.BCEcls(pcls, t)  # cls loss

            obji = self.BCEobj(pi[:, 4], tobj)
            loss[1] += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / self.anchors[i]  # wh ratio
                j = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            b, c = bc.long().T  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            tcls.append(c)  # class

        return tcls, tbox, indices
"""

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        #det = model.module.model[-1] if de_parallel(model) else model.model[-1]  # Detect() module
        m = de_parallel(model).model[-1]  # Detect() module
        self.stride = m.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEtheta = BCEtheta
        #self.nl = m.nl  # number of layers
        #self.nc = m.nc  # number of classes
        #self.anchors = m.anchors
        self.device = device
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(m, k))
        

    def __call__(self, p, targets):  # predictions, targets, model
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels])

        Return：
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, ltheta)).detach(): [4]
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors, tgaussian_theta = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # featuremap pixel
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                class_index = 5 + self.nc
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE
                
                # theta Classification by Circular Smooth Label
                t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                ltheta += self.BCEtheta(ps[:, class_index:], t_theta)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        return (lbox + lobj + lcls + ltheta) * bs, torch.cat((lbox, lobj, lcls, ltheta)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel

        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss()
        na = self.na
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # ttheta, tgaussian_theta = [], []
        tgaussian_theta = []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device).float()  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # targets (tensor): (n_gt_all_batch, c) -> (na, n_gt_all_batch, c) -> (na, n_gt_all_batch, c+1)
        # targets (tensor): (na, n_gt_all_batch, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index]])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            #anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            # theta = t[:, 6]
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # ttheta.append(theta) # theta, θ∈[-pi/2, pi/2)
            tgaussian_theta.append(gaussian_theta_labels)

        # return tcls, tbox, indices, anch
        return tcls, tbox, indices, anch, tgaussian_theta #, ttheta

class ComputeLoss_NEW:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEtheta = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['theta_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        m = de_parallel(model).model[-1]  # Detect() module
        
        # Check if anchors exist in the detection layer
        if hasattr(m, 'anchors'):
            anchors = m.anchors.clone().to(device)  # Copy anchors to the appropriate device
            #print(f"Anchors initialized: {anchors}")  # Debugging anchor initialization
            #print(f"Anchors shape: {anchors.shape}")
        else:
            raise AttributeError("Model does not contain anchors")
        
        
        self.stride = m.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        #self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.BCEtheta = BCEtheta
        #self.nc = m.nc  # number of classes
        #self.nl = m.nl  # number of layers
        #self.anchors = m.anchors.clone().to(device)
        self.device = device
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(m, k))
        #self.BCE_base = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, p, targets):  # predictions, targets
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels])

        Return：
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, ltheta)).detach(): [4]
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ltheta = torch.zeros(1, device=device)
        #tcls, tbox, indices = self.build_targets(p, targets)  # targets
        tcls, tbox, indices, anchors, tgaussian_theta = self.build_targets(p, targets)  # targets

        bs = p[0].shape[0]  # batch size
        n_labels = targets.shape[0]  # number of labels
        #loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses
        
        # Compute all losses
        #all_loss = []
        for i, pi in enumerate(p):  # layer index, layer predictions
            #b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)
                
                """
                pxy, pwh, pobj, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 2)  # target-subset of predictions
                # Regression
                pbox = torch.cat((pxy.sigmoid() * 1.6 - 0.3, (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]), 2)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(predicted_box, target_box)
                obj_target = iou.detach().clamp(0).type(pi.dtype)  # objectness targets

                all_loss.append([(1.0 - iou) * self.hyp['box'],
                                 self.BCE_base(pobj.squeeze(), torch.ones_like(obj_target)) * self.hyp['obj'],
                                 self.BCE_base(pcls, F.one_hot(tcls[i], self.nc).float()).mean(2) * self.hyp['cls'],
                                 obj_target,
                                 tbox[i][..., 2] > 0.0])  # valid
                """
                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] # featuremap pixel
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                #print("pbox: ", pbox)
                #print("pbox.shape: ", pbox.shape)
                #print("tbox[i]: ", tbox[i])
                #print("tbox[i].shape: ", tbox[i].shape)
                iou = bbox_iou(pbox.T, tbox[i], xywh=True, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                class_index = 5 + self.nc
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                    lcls += self.BCEcls(ps[:, 5:class_index], t)  # BCE
                
                # theta Classification by Circular Smooth Label
                t_theta = tgaussian_theta[i].type(ps.dtype) # target theta_gaussian_labels
                ltheta += self.BCEtheta(ps[:, class_index:], t_theta)
            
            
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ltheta *= self.hyp['theta']
        bs = tobj.shape[0]  # batch size
                
        '''
        # Lowest 3 losses per label
        n_assign = 4  # top n matches
        cat_loss = [torch.cat(x, 1) for x in zip(*all_loss)]
        ij = torch.zeros_like(cat_loss[0]).bool()  # top 3 mask
        sum_loss = cat_loss[0] + cat_loss[2]
        for col in torch.argsort(sum_loss, dim=1).T[:n_assign]:
            # ij[range(n_labels), col] = True
            ij[range(n_labels), col] = cat_loss[4][range(n_labels), col]
        loss[0] = cat_loss[0][ij].mean() * self.nl  # box loss
        loss[2] = cat_loss[2][ij].mean() * self.nl  # cls loss

        # Obj loss
        for i, (h, pi) in enumerate(zip(ij.chunk(self.nl, 1), p)):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # obj
            if n_labels:  # if any labels
                tobj[b[h], gj[h], gi[h]] = all_loss[i][3][h]
            loss[1] += self.BCEobj(pi[:, 4], tobj) * (self.balance[i] * self.hyp['obj'])
        '''
        #return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses
        return (lbox + lobj + lcls + ltheta) * bs, torch.cat((lbox, lobj, lcls, ltheta)).detach()

    def build_targets(self, p, targets):
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) pixel

        Return：non-normalized data
            tcls (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
            tbox (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 4) featuremap pixel
            indices (list[P3_out,...]): len=self.na, tensor.size(4, n_filter2) [b, a, gj, gi]
            anch (list[P3_out,...]): len=self.na, tensor.size(n_filter2, 2)
            tgaussian_theta (list[P3_out,...]): len=self.na, tensor.size(n_filter2, hyp['cls_theta'])
            # ttheta (list[P3_out,...]): len=self.na, tensor.size(n_filter2)
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na = self.na
        nt = targets.shape[0]  # number of anchors, targets
        #tcls, tbox, indices = [], [], []
        tcls, tbox, indices, anch = [], [], [], []
        tgaussian_theta = []
        #gain = torch.ones(6, device=self.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device).float()  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        '''
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float()  # offsets
        '''
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]  # anchors
            #print(f'Type of p[{i}] is: {type(p[i])}')  # Debug: Check the type of p[i]
            #feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]
            #feature_wh[0:2] = torch.tensor([p[i][0].shape[3], p[i][0].shape[2]], device=self.device)  # [w_f, h_f]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]
            #print(f'feature_wh[0:2] is: {feature_wh[0:2]}')  # Debug: Check the feature_wh[0:2]

            # Match targets to anchors
            #t = targets * gain  # shape(3,n,7)
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            
            if nt:
                # # Matches
                #r = t[..., 4:6] / self.anchors[i]  # wh ratio
                #a = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # a = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[a]  # filter
                
                #print(anchors)
                
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)
                
                # # Offsets
                gxy = t[:, 2:4]  # grid xy
                #gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                #t = t.repeat((5, 1, 1))
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                #offsets = torch.zeros_like(gxy)[None] + off[:, None]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
                #t[..., 4:6][~j] = 0.0  # move unsuitable targets far away
            else:
                t = targets[0]
                offsets = 0

            # Define t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            '''
            bc, gxy, gwh = t.chunk(3, 2)  # (image, class), grid xy, grid wh
            b, c = bc.long().transpose(0, 2).contiguous()  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.transpose(0, 2).contiguous()  # grid indices
            '''
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T.long()  # grid xy indices
            
            
            #print(f"gi type: {gi.dtype}, gj type: {gj.dtype}")
            #print(f"feature_wh[0] type: {feature_wh[0].dtype}, feature_wh[1] type: {feature_wh[1].dtype}")
            # Convert feature_wh values to int for compatibility
            gi = gi.clamp(0, int(feature_wh[0].item() - 1))
            gj = gj.clamp(0, int(feature_wh[1].item() - 1))
            
            # Append
            a = t[:, -1].long()  # anchor indices 取整
            #indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            #indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, int(feature_wh[1].item()) - 1), gi.clamp_(0, int(feature_wh[0].item()) - 1)))  # image, anchor, grid indices
            #tbox.append(torch.cat((gxy - gij, gwh), 2).permute(1, 0, 2).contiguous())  # box
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tgaussian_theta.append(gaussian_theta_labels)

            # # Unique
            # n1 = torch.cat((b.view(-1, 1), tbox[i].view(-1, 4)), 1).shape[0]
            # n2 = tbox[i].view(-1, 4).unique(dim=0).shape[0]
            # print(f'targets-unique {n1}-{n2} diff={n1-n2}')

        #return tcls, tbox, indices
        return tcls, tbox, indices, anch, tgaussian_theta #, ttheta