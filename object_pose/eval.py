"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import os
import numpy as np
import torch
import network_bingham
from utils import qrotate_pc, qconjugate, qmult
from pytorch3d.loss import chamfer_distance

num_point = 2048
batch_size = 32
device = torch.device('cuda:0')

nm = 50

net = network_bingham.MBN(num_point, 3, 128, nm)
net = net.to(device)


def eval_cls(cls):
    ## change the weight_fn to the expected one
    weight_fn = 'log_{}/chkpt.pth'.format(cls)
    if not os.path.exists(weight_fn):
        print('{} not exists.'.format(weight_fn))
        return

    print('Initializing network')

    state_dict = torch.load(weight_fn)
    print('loading weights from {}'.format(weight_fn))
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    print('Network initialization done')

    test_data_fn = './data/benchmark/{}.npy'.format(cls)
    test_data = np.load(test_data_fn, allow_pickle=True)

    cd_lst = []
    for idx, (pc, gt_q) in enumerate(test_data):
        points = torch.from_numpy(pc).float().to(device).reshape(1, num_point, pc.shape[1])
        gt_q = torch.from_numpy(gt_q).float().to(device).reshape(1, 4)

        pred_q, pred_l, weights = net(points)

        rel_q = qmult(pred_q, qconjugate(gt_q))

        rel_q_tiled = rel_q.reshape(nm, 1, 4).repeat(1, pc.shape[0], 1).reshape(-1, 4)
        points_tiled = points.reshape(1, pc.shape[0], 3).repeat(nm, 1, 1).reshape(-1, 3)

        rotated_pc = qrotate_pc(points_tiled, rel_q_tiled)
        rotated_pc = rotated_pc.reshape(nm, pc.shape[0], 3)

        dists = chamfer_distance(points_tiled.reshape(nm, pc.shape[0], 3), rotated_pc, batch_reduction=None)[0]
        best_dist = dists[weights.argmax()].item()

        cd_lst.append(best_dist)

    print('{}: {}'.format(cls, np.mean(cd_lst)))


def eval_all():
    classes = [
        'bathtub',
        'bed',
        'chair',
        'desk',
        'dresser',
        'monitor',
        'night_stand',
        'sofa',
        'table',
        'toilet',
    ]

    for cls in classes:
        eval_cls(cls)


if __name__ == '__main__':
    eval_all()
