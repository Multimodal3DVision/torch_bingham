"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""


import os.path as osp
import numpy as np
from torch.utils import data
from torchvision.datasets.folder import default_loader
import os


class AmbiguousRelocData(data.Dataset):
    """
    pytorch data loader for the ambiguous scene dataset
    """
    def __init__(self, scene, data_path, train, transform=None,
                 seed=0):
        """
        :param scene: scene name
        :param data_path: root data directory
        :param train: if True, return the training images. If False, returns the
        testing images
        :param transform: transform to apply to the images
        """
        self.transform = transform
        np.random.seed(seed)

        base_dir = osp.join(osp.expanduser(data_path), scene)

        self.imgs = []
        self.poses = []

        if train:
            base_dir = osp.join(base_dir, 'train')
        else:
            base_dir = osp.join(base_dir, 'test')


        for seq in os.listdir(base_dir):
            # read poses and collect image names
            seq_dir = osp.join(base_dir, seq)
            filenames = [n for n in os.listdir(osp.join(seq_dir, 'rgb_matched')) if
                           n.find('frame') >= 0]
            filenames = np.sort(filenames)
            poses_Rt = self.read_poses(osp.join(seq_dir, 'poses_%s.txt' % seq))

            self.poses.extend(poses_Rt)

            imgs = [osp.join(seq_dir, 'rgb_matched', img_file) for img_file in filenames]
            self.imgs.extend(imgs)
        self.poses = np.asarray(self.poses)

    def read_poses(self, path):
        """
        Reads poses from a comma separated text file. Each line is assumed to consist of one pose,
        represented by a quaternion and a translation.
        To remove ambiguity, quaternions are mapped to lie on only one hemisphere.
        :param path: path to pose file
        :return Nx7 pose array
        """
        file = open(path)
        poses = []

        for line in file:
            pose = np.zeros([7], dtype=np.float32)
            line = line.rstrip('\r\n')
            line = line.split(',')
            q = np.array([float(line[2]), float(line[3]), float(line[4]),
                       float(line[5])])
            if np.sign(q[0]) != 0:
                q *= np.sign(q[0])

            pose[:4] = q
            pose[4] = float(line[6])
            pose[5] = float(line[7])
            pose[6] = float(line[8])
            poses.append(pose)
        return np.asarray(poses)

    def __getitem__(self, index):
        img = None
        pose = None
        # get next image and pose label
        while img is None:
            img = default_loader(self.imgs[index])
            pose = self.poses[index]
            index += 1
        index -= 1

        # apply specified transformation to the image
        if self.transform is not None:
            img = self.transform(img)

        return img, pose

    def __len__(self):
        return self.poses.shape[0]

