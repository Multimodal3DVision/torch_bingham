"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

import torch
from torch import nn
import numpy as np
import torch_bingham
import torch.nn.functional as F


class mixture_bingham(torch.nn.Module):
    """
    Bingham Loss optimizing negative Log Likelihood
    """
    def __init__(self):
        super(mixture_bingham, self).__init__()
        self.loss_bingham = bingham_log_likelihood()

    def forward(self, predicted_rotation, predicted_lambda, predicted_coeff, target):
        num_coeff = predicted_coeff.size()[1]

        if num_coeff > 1:

            predicted_rotation = predicted_rotation.reshape(-1, 4)
            predicted_Lambda = predicted_lambda.reshape(-1, 3)
            target = target.reshape(-1, 1, 4).repeat([1, num_coeff, 1]).reshape(-1, 4)
            losses = self.loss_bingham(predicted_rotation, predicted_lambda, target)
            losses = losses.reshape(-1, num_coeff)

            log_pdf = losses + torch.log(predicted_coeff)
            mixture_loss = -torch.mean(torch.logsumexp(log_pdf, dim=1))

        else:
            loglikelihood = self.loss_bingham(predicted_rotation, predicted_lambda, target)
            mixture_loss = -torch.mean(loglikelihood)

        return mixture_loss.view([1])


class bingham_log_likelihood(torch.nn.Module):
    """
    Bingham Log Likelihood
    """
    def __init__(self):
        super(bingham_log_likelihood, self).__init__()

    def forward(self, predicted_rotation, predicted_lambda, target):

        quaternion = predicted_rotation.reshape(-1,4)
        log_p = torch_bingham.bingham_prob(quaternion, predicted_lambda, target)
        return log_p


class gauss_log_likelihood(torch.nn.Module):
    """
    Gaussian Log Likelihood
    """
    def __init__(self):
        super(gauss_log_likelihood, self).__init__()

    def forward(self, predicted_translation, variance, target):
        variance = variance + 1e-8
        dim = predicted_translation.size()[1]
        
        x = ((target - predicted_translation)**2)
        x = torch.sum(x / (variance), dim=1)
        det = 1.0
        for i in range(dim):
            det *= variance[:,i]
        log_p = -0.5*(x + dim * torch.log(torch.tensor((2 * np.pi))) +
                      torch.log(det))
        return log_p

class mixture_gauss(torch.nn.Module):
    """
    Gaussian Loss optimizing negative Log Likelihood
    """
    def __init__(self):
        super(mixture_gauss, self).__init__()
        self.loss_gauss = gauss_log_likelihood()

    def forward(self, predicted_translation, predicted_var, predicted_coeff, target):
        num_coeff = predicted_coeff.size()[1]
        dim = int(predicted_var.size()[1] / num_coeff)
        if num_coeff > 1:

            predicted_translation = predicted_translation.reshape(-1, dim)
            predicted_vars = predicted_var.reshape(-1, dim)
            target = target.reshape(-1, 1, dim).repeat([1, num_coeff, 1]).reshape(-1, dim)
            losses = self.loss_gauss(predicted_translation, predicted_vars, target)
            losses = losses.reshape(-1, num_coeff)

            log_pdf = losses + torch.log(predicted_coeff)
            mixture_loss = -torch.mean(torch.logsumexp(log_pdf, dim=1))

        else:
            loglikelihood = self.loss_gauss(predicted_translation, predicted_var, target)
            mixture_loss = -torch.mean(loglikelihood)


        return mixture_loss.view([1])

class rWTALoss(nn.Module):
    """
    RWTA Loss with Bingham and Gaussian Mixture Models
    """

    def __init__(self, num_h, epsilon=0.95):
        super(rWTALoss, self).__init__()
        self.num_h = num_h
        self.loss_b = bingham_log_likelihood()
        self.loss_g = gauss_log_likelihood()
        self.epsilon = epsilon


    def forward(self, pred_q, pred_l, weights, gt_q, pred_x, pred_var, gt_t):
        dZ = pred_l.reshape(-1, 3)
        pred_q = pred_q.reshape(-1, 4)

        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.num_h, 1]).reshape(-1, 4)
        pred_var = pred_var.reshape(-1, 3)

        pred_x = pred_x.reshape(-1, 3)
        gt_t = gt_t.reshape(-1, 1, 3).repeat([1, self.num_h, 1]).reshape(-1, 3)

        # choose best branch
        l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.num_h)
        l2 = torch.abs((pred_x - gt_t)).sum(-1).reshape(-1, self.num_h)
        l1 = l1 + l2
        best_indices = l1.argmin(1)

        # bingham loss
        p = self.loss_b(pred_q, dZ, gt_q)
        p = p.reshape([-1, self.num_h])
        all_assignments = torch.mean(-p)
        best_assignment = torch.mean(-p[torch.arange(p.shape[0]), best_indices])

        # mixture coefficient loss
        weight_loss = F.cross_entropy(weights, best_indices)
        loss = (self.epsilon - 1 / self.num_h) * best_assignment + (
                                    1 - self.epsilon) * 1 / self.num_h * all_assignments\

        # gauss loss
        p = self.loss_g(pred_x, pred_var, gt_t)
        p = p.reshape([-1, self.num_h])
        all_assignments = torch.mean(-p)
        best_assignment = torch.mean(-p[torch.arange(p.shape[0]), best_indices])

        gloss = (self.epsilon - 1 / self.num_h) * best_assignment + (
                1 - self.epsilon) * 1 / self.num_h * all_assignments

        return loss, weight_loss, gloss

class eWTALoss(nn.Module):
    """
    EWTA Loss with Bingham and Gaussian Mixture Models
    """
    def __init__(self, num_h):
        super(eWTALoss, self).__init__()
        self.num_h = num_h
        self.loss_b = bingham_log_likelihood()
        self.loss_g = gauss_log_likelihood()


    def forward(self, pred_q, pred_l, weights, gt_q, pred_x, pred_var, gt_t, k):
        dZ = pred_l.reshape(-1, 3)
        pred_q = pred_q.reshape(-1, 4)

        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.num_h, 1]).reshape(-1, 4)
        pred_std = pred_var.reshape(-1, 3)

        pred_x = pred_x.reshape(-1, 3)
        gt_t = gt_t.reshape(-1, 1, 3).repeat([1, self.num_h, 1]).reshape(-1, 3)

        l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.num_h)
        l2 = torch.abs((pred_x - gt_t)).sum(-1).reshape(-1, self.num_h)
        l1 = l1 + l2
        best_indices = torch.argsort(l1, dim=1)
        topk = best_indices[:, :k]

        topkb = torch.zeros_like(weights)
        topkb = topkb.scatter_(1, topk, 1.)
        p = self.loss_b(pred_q, dZ, gt_q)
        p = p.reshape([-1, self.num_h])
        best_assignment = torch.mean(-torch.gather(p, 1, topk).sum(1))
        loss = best_assignment

        weight_loss = F.binary_cross_entropy_with_logits(weights, topkb)

        p = self.loss_g(pred_x, pred_std, gt_t)
        p = p.reshape([-1, self.num_h])
        best_assignment = torch.mean(-torch.gather(p, 1, topk).sum(1))
        gloss = best_assignment

        return torch.mean(loss), weight_loss, gloss
