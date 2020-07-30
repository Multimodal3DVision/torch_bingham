"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import torch
import torch_bingham
from torch import nn
from torch.nn import functional as F


class RWTA_MB_Loss(nn.Module):
    def __init__(self, nm, epsilon=0.95, use_l1=False):
        super(RWTA_MB_Loss, self).__init__()
        self.nm = nm
        self.epsilon = epsilon
        self.use_l1 = use_l1

    def forward(self, pred_q, Zbatch, weights, gt_q):
        pred_q = pred_q.reshape(-1, 4)
        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.nm, 1]).reshape(-1, 4)

        p = torch_bingham.bingham_prob(pred_q, Zbatch, gt_q)
        p = p.reshape([-1, self.nm])

        if self.use_l1:
            l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.nm)
            best_indices = l1.argmin(1)
        else:
            best_indices = p.argmax(1)

        all_assignments = - torch.mean(p)
        best_assignment = - torch.mean(p[torch.arange(p.shape[0]), best_indices])

        rwta_loss = (self.epsilon - 1 / self.nm) * best_assignment + (
                1 - self.epsilon) * 1 / self.nm * all_assignments

        weights = F.softmax(weights, dim=-1)

        mb_loss = -torch.mean(torch.logsumexp(torch.log(weights) + p, dim=-1))

        return rwta_loss, mb_loss


class RWTA_CE_Loss(nn.Module):
    def __init__(self, nm, epsilon=0.95, use_l1=False):
        super(RWTA_CE_Loss, self).__init__()
        self.nm = nm
        self.epsilon = epsilon
        self.use_l1 = use_l1

    def forward(self, pred_q, Zbatch, weights, gt_q):
        pred_q = pred_q.reshape(-1, 4)
        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.nm, 1]).reshape(-1, 4)

        p = torch_bingham.bingham_prob(pred_q, Zbatch, gt_q)
        p = p.reshape([-1, self.nm])

        if self.use_l1:
            l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.nm)
            best_indices = l1.argmin(1)
        else:
            best_indices = p.argmax(1)

        all_assignments = - torch.mean(p)
        best_assignment = - torch.mean(p[torch.arange(p.shape[0]), best_indices])

        rwta_loss = (self.epsilon - 1 / self.nm) * best_assignment + (
                1 - self.epsilon) * 1 / self.nm * all_assignments

        ce_loss = F.cross_entropy(weights, best_indices)

        return rwta_loss, ce_loss
