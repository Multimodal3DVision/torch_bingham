"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

import numpy as np

def compute_rotation_error(gt, rotation):
    """Function to compute the angular error between predicted and ground truth rotation

    Parameters
    ----------
    :param gt: ground truth quaternion
    :param rotation: predicted rotation as quaternion

    :return difference in orientation

    """
    q = rotation
    if q[0] < 0:
        q = -1 * q
    d = np.abs(np.dot(q, gt[0:4]))
    d = np.minimum(1.0, d)
    angle_error = 2 * np.arccos(d)*180 / np.pi

    return angle_error

def evaluate_pose_batch(gt, rotations, translations):
    """Function to evaluate predicted poses in comparison to the ground truth ones

    Parameters
    ----------
    :param gt: ground truth pose given vector [N, q, t] (quaternion and translation)
    :param rotation: predicted rotations as quaternion [N, 4]
    :param translation: predicted translations [N, 3]

    :return difference in rotation (angle) and translation (||gt - predicted_t||_2)

    """
    assert (gt.shape[0] == rotations.shape[0]), \
        'batch size of ground truth and prediction must be equal'

    angle_errors = np.zeros([gt.shape[0], 1])
    for i in range(0, gt.shape[0]):
        angle_errors[i] = compute_rotation_error(gt[i,:4], rotations[i])
    translation_errors = np.linalg.norm(translations - gt[:,4:7], axis=1)
    return angle_errors, translation_errors

def compute_pose_histogram(thresholds, predicted_rotations, predicted_translations, gts):
    """Function to compute the percentage of poses below given pre-defined thresholds

    Parameters
    ----------
    :param thresholds: list with tuples of thresholds (orientation, translation)
    :param predicted_rotations: predicted rotation as quaternion
    :param predicted_translations: predicted translation


    :return list of percentages for the defined thresholds
    """
    hist = np.zeros(len(thresholds))
    gts = np.asarray(gts)
    predicted_rotations = np.asarray(predicted_rotations)
    predicted_translations = np.asarray(predicted_translations)

    rotation_errors, translation_errors = evaluate_pose_batch(gts, predicted_rotations, predicted_translations)

    for i in range(len(rotation_errors)):
        for j in range(len(thresholds)):
            rotation_threshold, translation_threshold = thresholds[j]
            if (rotation_errors[i] <= rotation_threshold) and (translation_errors[i] <= translation_threshold):
                hist[j] +=1

    return hist / gts.shape[0]

