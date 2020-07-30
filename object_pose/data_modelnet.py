"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import os
import numpy as np
from transforms3d import quaternions as tq
from torch.utils.data import Dataset


def load_modelnet10(cls='table', split='train'):
    data_fn = './data/modelnet_{}_{}.npz'.format(cls, split)
    if os.path.exists(data_fn):
        print('loading data from {}'.format(data_fn))
        return np.load(data_fn)['pc']
    else:
        print('{} does not exist.'.format(data_fn))
        return None


class PointDownsample(object):
    def __init__(self, npoints=2048):
        self.npoints = npoints

    def __call__(self, points):
        return points[np.random.choice(len(points), self.npoints)]


class ModelNetDatasetIcoshpere(Dataset):
    def __init__(self, data, transforms, config='oim06'):
        self.data = data
        self.len = data.shape[0]
        self.transforms = transforms
        self.quats = load_sample_poses(config)
        self.n_q = len(self.quats)
        print('Generating {} quats'.format(self.n_q))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pc = self.data[idx]
        if self.transforms is not None:
            pc = self.transforms(pc)

        r_idx = np.random.randint(0, self.n_q)
        q = self.quats[r_idx]
        q = (2 * (q[0] > 0) - 1) * q
        rot = tq.quat2mat(q)
        if pc.shape[1] == 3:
            t_pc = pc @ rot.T
        elif pc.shape[1] == 6:
            t_ver = pc[:, :3] @ rot.T
            t_nor = pc[:, 3:6] @ rot.T
            t_pc = np.concatenate([t_ver, t_nor], axis=1)
        return t_pc, q


def init_rot_mat(config_fn):
    data = np.loadtxt(config_fn)
    phi = data[:, 0]
    theta = data[:, 1]
    psi = data[:, 2]
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    rMat = np.zeros((data.shape[0], 9))

    rMat[:, 0] = cpsi * cphi - spsi * ctheta * sphi
    rMat[:, 1] = -cpsi * sphi - spsi * ctheta * cphi
    rMat[:, 2] = spsi * stheta

    rMat[:, 3] = spsi * cphi + cpsi * ctheta * sphi
    rMat[:, 4] = -spsi * sphi + cpsi * ctheta * cphi
    rMat[:, 5] = -cpsi * stheta

    rMat[:, 6] = stheta * sphi
    rMat[:, 7] = stheta * cphi
    rMat[:, 8] = ctheta

    rMat = rMat.reshape(-1, 3, 3)

    q_list = []
    for m in rMat:
        q = tq.mat2quat(m)
        q = (2 * (q[0] > 0) - 1) * q
        q_list.append(q)
    q_list = np.stack(q_list)
    return q_list


def load_sample_poses(config='oim06'):
    config_fn = './rotation_samples/{}.eul'.format(config)
    q_list = init_rot_mat(config_fn)
    return q_list
