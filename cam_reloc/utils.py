"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

import numpy as np

def close_to_all_pose(samples, gt):
    """
    Function to retrieve the closest pose to the ground truth

    Parameters
    ----------
    :param samples: pose predictions
    :param gt: ground truth pose

    :return closest pose and index of it
    """

    diff_t = np.zeros([samples.shape[0], samples.shape[0]])
    for i in range(0, samples.shape[0]):
        for j in range(0, samples.shape[0]):
            diff_t[i, j] = np.linalg.norm(samples[i, 4:] - gt[j, 4:])

    dot = np.minimum(np.matmul(samples[:, :4], gt[:, :4].T), 1.0)
    diff_r = np.rad2deg(2 * np.arccos(np.abs(dot)))

    sum_r = np.sum(diff_r, axis=1)
    sum_t = np.sum(diff_t, axis=1)
    sums = (sum_r / np.max(sum_r)) + (sum_t / np.max(sum_t))
    idx = np.argmin(sums)
    return samples[idx], idx

