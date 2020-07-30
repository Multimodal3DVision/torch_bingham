"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

import torch.optim as optim
import torch


import os
import Losses
import eval as ev
import numpy as np
from CamPoseNet import CamPoseNet
import utils as utils
from torch.optim.lr_scheduler import ExponentialLR
import torch_bingham

class CamPose():

    def __init__(self, args, device, pretrained=False):
        super(CamPose, self).__init__()
        self.args = args
        self.exp = '{}{}_{}_{}_numcoeff{}/'.format(self.args.save_dir,
                    self.args.dataset, self.args.scene,
                    self.args.base, str(self.args.num_coeff))

        print(self.exp)
        self.losses = {}
        self.device = device
        self.model = CamPoseNet(args.num_coeff, args.base, pretrained)

        #self.losses['RPose'] = Losses.mixture_bingham()
        #self.losses['TPose'] = Losses.mixture_gauss()

        self.losses['PoseRWTA'] = Losses.rWTALoss(self.args.num_coeff)
        #self.losses['PoseEWTA'] = Losses.eWTALoss(self.args.num_coeff)

        self.model.to(device)
        if args.stage == 1:
            to_optimize = list(self.model.fc_pose.parameters()) + list(self.model.fc_Z.parameters())
        else:
            to_optimize = self.model.parameters()

        self.optimizer = optim.Adam(to_optimize, lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, 0.7)

        if args.restore:
            self.model.load_state_dict(torch.load(self.exp + args.model), strict=False)
            print('Restored model')
        print('Initialized')

    def train(self, train_loader):

        for epoch in range(self.args.num_epochs):

            # gradually decrease the number of hypothesis for EWTA training
            #k = int(np.floor((1 - (epoch + 1.0) /
            #                  self.args.num_epochs) * (self.args.num_coeff * 1.0))) + 1
            #print("Best %f used" % k)

            for batch_idx, (data, target) in enumerate(train_loader):

                self.model.train()
                self.optimizer.zero_grad()

                inputs = data.to(self.device)

                outputs = self.model(inputs, True)
                target = target.to(self.device)

                #accR = self.losses['RPose'](outputs[0], outputs[1], outputs[2], target[:, :4])
                #accT = self.losses['TPose'](outputs[3], outputs[4], outputs[2], target[:, 4:7])

                accR, wloss, accT = self.losses['PoseRWTA'](outputs[0], outputs[1], outputs[2],
                                                        target[:, 0:4], outputs[3], outputs[4], target[:, 4:7])

                #accR, wloss, accT = self.losses['PoseEWTA'](outputs[0], outputs[1], outputs[2],
                #                                        target[:, 0:4], outputs[3], outputs[4], target[:, 4:7], k)

                if self.args.stage == 0:
                    acc = accT
                elif self.args.stage == 1:
                    acc = accR
                else:
                    acc = accT + accR + wloss

                acc.backward()
                self.optimizer.step()

                print("Epoch: [%2d] [%4d], training error %g" % (epoch, batch_idx, acc.item()))

                self.model.eval()

            outputs = [outputs[i].to('cpu').data.numpy() for i in range(len(outputs))]
            target = target.to('cpu').data.numpy()


            pred_r,_, pred_t,_ = self.extract_predictions(outputs, target, type=self.args.prediction_type)
            pred_r_oracle,_, pred_t_oracle,_ = self.extract_predictions(outputs, target, type='oracle')

            oracle_rotation_error, oracle_translation_error = ev.evaluate_pose_batch(target, pred_r_oracle,
                                                                         pred_t_oracle)

            print(("(Oracle) Median rotation error %f, translation error %f, %f, %f") % (
            np.median(oracle_rotation_error), np.median(oracle_translation_error), np.mean(oracle_translation_error), np.std(oracle_translation_error)))

            rotation_error, translation_error = ev.evaluate_pose_batch(target, pred_r,pred_t)
            print(("Median rotation error %f, translation error %f, %f, %f") % (np.median(rotation_error),
                                                                np.median(translation_error), np.mean(translation_error), np.std(translation_error)))

            if epoch > 0 and epoch % 50 == 0:
                self.scheduler.step()

        if not os.path.exists(self.exp):
            os.makedirs(self.exp)
        torch.save(self.model.state_dict(), self.exp + 'model_%d' % (epoch))
        torch.save(self.optimizer.state_dict(), self.exp + 'optimizer')

    def eval(self, loader):
        results = {}
        results['rotations'] = []
        results['rotations_best'] = []
        results['translations'] = []
        results['translations_best'] = []
        results['labels'] = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                self.model.eval()
                inputs = data.to(self.device)
                outputs = self.model(inputs)
                outputs = [outputs[i].to('cpu').numpy() for i in range(0, len(outputs))]
                target = target.to('cpu').numpy()

                pred_r, lambdas, pred_t, pred_vars = self.extract_predictions(outputs, target, type=self.args.prediction_type)
                pred_r_oracle, lambdas_oracle, pred_t_oracle, pred_vars_oracle = self.extract_predictions(outputs, target, type='oracle')

                # store everything for evaluation
                results['rotations'].extend(pred_r)
                results['rotations_best'].extend(pred_r_oracle)
                results['translations'].extend(pred_t)
                results['translations_best'].extend(pred_t_oracle)
                results['labels'].extend(target)

        return results

    def gauss_entropies(self, pred_t, pred_var):
        entropies = np.zeros([pred_t.size()[0]])
        for i in range(0, pred_t.size()[0]):
            m = torch.distributions.MultivariateNormal(pred_t[i], torch.eye(3) * pred_var[i])
            entropies[i] = m.entropy().numpy()
        return entropies

    def bingham_entropies(self, lambdas):
        entropies = torch_bingham.bingham_entropy(lambdas)
        return entropies

    def extract_predictions(self, outputs, target, type=None):
        batch_size = target.shape[0]
        predicted_rotation = outputs[0].reshape(-1, self.args.num_coeff, 4)
        predicted_lambda = outputs[1].reshape(-1, self.args.num_coeff, 3)
        predicted_translation = outputs[3].reshape(-1, self.args.num_coeff, 3)
        predicted_var = outputs[4].reshape(-1, self.args.num_coeff, 3)

        coeffs = outputs[2]

        predicted_ts = np.zeros([batch_size, 3])
        predicted_rs = np.zeros([batch_size, 4])
        predicted_lambdas = np.zeros([batch_size, 3])
        predicted_vars = np.zeros([batch_size, 3])

        if type == 'highest':
            # pose from one model with largest coefficient
            index = np.asarray(np.argsort(coeffs, axis=1), dtype=np.int16)
            for i in range(0, batch_size):

                predicted_rs[i] = predicted_rotation[i, index[i, -1]]
                predicted_lambdas[i] = predicted_lambda[i, index[i, -1]]
                predicted_ts[i] = predicted_translation[i, index[i, -1]]
                predicted_vars[i] = predicted_var[i, index[i, -1]]

        else:
            # Oracle prediction
            gts = np.tile(target, (self.args.num_coeff, 1, 1))
            predicted_rt = np.concatenate([predicted_rotation, predicted_translation], axis=2)

            for i in range(0, batch_size):
                best, idx = utils.close_to_all_pose(predicted_rt[i], gts[:, i, :])
                predicted_rs[i] = best[:4]
                predicted_lambdas[i] = predicted_lambda[i, idx]
                predicted_ts[i] = best[4:]
                predicted_vars[i] = predicted_var[i, idx]

        return predicted_rs, predicted_lambdas, predicted_ts, predicted_vars

