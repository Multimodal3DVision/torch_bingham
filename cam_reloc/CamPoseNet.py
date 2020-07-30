"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CamPoseNet(nn.Module):

    def __init__(self, num_coeff, base='ResNet', pretrained=False):
        """Initialized the network architecture.

            Parameters
            ----------
            :param num_coeff : the number of components in the mixture model
            :param base: the network architecture to use, currently 'ResNet' or 'Inception'
            :param pretrained: True if a pretrained network from Pytorch should be used, False otherwise

            Raises
            ------
            NotImplementedError
                If a specific network architecture is not implemented.
        """

        super(CamPoseNet, self).__init__()
        self.num_coeff = num_coeff
        self.base = base

        if self.base == 'ResNet':
            self.model = models.resnet34(pretrained=pretrained)
            fe_out_planes = self.model.fc.in_features
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(fe_out_planes, 2048)

        elif self.base == 'Inception':
            self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
            self.model.fc = nn.Linear(2048, 2048)
        else:
            raise NotImplemented

        self.fc_xyz = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))
        self.fc_pose = nn.Sequential(nn.Linear(2048, 4 * self.num_coeff))

        self.fc_Z = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))
        self.fc_coeff = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                            nn.Linear(512, self.num_coeff), nn.BatchNorm1d(self.num_coeff))
        self.fc_std = nn.Sequential(nn.Linear(2048, 3 * self.num_coeff))

        # initialize layers
        if pretrained:
            init_modules = [self.fc_pose, self.fc_xyz, 
                            self.fc_Z, self.fc_std, self.fc_coeff]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)



    def forward(self, x, train=False):
        x = self.model(x)
        x = F.relu(x)
       
        wpqr = self.fc_pose(x)
        xyz = self.fc_xyz(x)
       
        # quaternion normalization
        nwpqr = wpqr.reshape(-1, self.num_coeff, 4)
        nwpqr = F.normalize(nwpqr, dim=2)
        nwpqr = nwpqr.reshape(-1, self.num_coeff * 4)

        Z = self.fc_Z(x)
        Z = F.softplus(Z)
        dZ = Z.reshape(-1,3)

        # compute ordered lambdas from predicted offsets
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zbatch = torch.cat([Z0, Z1, Z2], dim=1)
        lambdas = -1 * Zbatch.clamp(1e-12, 900)
        lambdas = lambdas.reshape(-1, self.num_coeff*3)

        std = self.fc_std(x)
        std = F.softplus(std)
        coeff = self.fc_coeff(x)

        #coeff = F.softmax(coeff, dim=1)
        coeff = F.relu(coeff)

        return [nwpqr, lambdas, coeff, xyz, std]


