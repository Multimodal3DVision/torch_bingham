"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import torch
from torch import nn
from torch.nn import functional as F


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bn=True,
                 activation_fn=nn.ReLU(inplace=True)):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups,
                              bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output


class dense(nn.Module):
    def __init__(self, in_features, out_features, bn=True, bias=True, activation_fn=nn.ReLU(inplace=True)):
        super(dense, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if bn:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = None

        if activation_fn:
            self.activation = activation_fn
        else:
            self.activation = None

    def forward(self, input):
        output = self.fc(input)
        if self.bn:
            output = self.bn(output)
        if self.activation:
            output = self.activation(output)
        return output


class MBN(nn.Module):
    def __init__(self, num_points, input_dim, feat_dim, nm, bn=True):
        super(MBN, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.nm = nm

        self.conv1 = conv2d(input_dim, 64, (1, 1), bn=bn)
        self.conv2 = conv2d(64, 128, (1, 1), bn=bn)
        self.conv3 = conv2d(128, 128, (1, 1), bn=bn)

        self.conv4 = conv2d(128, feat_dim, (1, 1), bn=bn)
        self.maxpool = nn.MaxPool2d((self.num_points, 1))

        self.rot_layers = nn.Sequential(*[
            dense(feat_dim, 256, bn=bn),
            dense(256, 256, bn=bn),
        ])

        self.q_layer = dense(256, nm * 4, activation_fn=False, bn=None)
        self.l_layer = dense(256, nm * 3, activation_fn=None, bn=None)
        self.weights = dense(256, nm, activation_fn=nn.Sigmoid(), bn=None)

    def forward(self, input):
        input_reshaped = input.reshape(-1, 1, self.num_points, self.input_dim)
        input_reordered = input_reshaped.permute(0, 3, 2, 1)

        conv1_output = self.conv1(input_reordered)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        max_pool_outptut = self.maxpool(conv4_output)
        max_pool_reshaped_output = max_pool_outptut.reshape(max_pool_outptut.shape[0], -1)
        rot_layers_output = self.rot_layers(max_pool_reshaped_output)
        q_layer_output = self.q_layer(rot_layers_output)
        l_layer_output = F.softplus(self.l_layer(rot_layers_output))

        # convert from original output of network to lambdas
        dZ = l_layer_output.reshape(-1, 3)
        Z0 = dZ[:, 0:1]
        Z1 = Z0 + dZ[:, 1:2]
        Z2 = Z1 + dZ[:, 2:3]
        Zbatch = torch.cat([Z0, Z1, Z2], dim=1)
        Zbatch = -1 * Zbatch.clamp(1e-12, 900)

        # normalize q
        q_layer_output = q_layer_output.reshape(-1, 4)
        norm_q_output = torch.norm(q_layer_output, dim=-1, keepdim=True)
        normalized_q_output = q_layer_output / (norm_q_output + 1e-12)
        normalized_q_output = ((normalized_q_output[:, 0:1] > 0).float() - 0.5) * 2 * normalized_q_output

        weights = self.weights(rot_layers_output)
        return normalized_q_output, Zbatch, weights
