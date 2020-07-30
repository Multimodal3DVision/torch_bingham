"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import torch


def qmult(q1, q2):
    # q1: N x 4
    # q2: N x 4
    w1, x1, y1, z1 = torch.split(q1, 1, dim=-1)
    w2, x2, y2, z2 = torch.split(q2, 1, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.cat([w, x, y, z], dim=-1)


def qconjugate(q):
    q1 = torch.cat([q[:, 0:1], q[:, 1:] * -1.0], dim=-1)
    return q1


def qrotate_pc(pc, q):
    # make sure it's either using one quaternion to rotate the whole pc
    # or rotate each pt with an individual quaternion
    assert q.shape[0] == 1 or pc.shape[0] == q.shape[0]
    pad = torch.zeros((pc.shape[0], 1), dtype=pc.dtype).to(pc.device)
    padded_pc = torch.cat([pad, pc], dim=1)
    return qmult(q, qmult(padded_pc, qconjugate(q)))[:, 1:]
