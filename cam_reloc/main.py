"""
   6d cam relocalization module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Mai Bui
"""

# Basic imports
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
import random
from CamPose import CamPose
import eval
from dataset_loaders.ambiguous_reloc_data import AmbiguousRelocData


seed = 13
print('Seed: %d' % seed)

np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ReLoc', help='Specify which dataset to use, "ReLoc" for our ambiguous scene dataset.')
parser.add_argument('--scene', default='meeting_table', help='Specify which scene to use, "seminar", "meeting table", ...')
parser.add_argument('--base_dir', default='./',help='Base directory to use.')
parser.add_argument('--save_dir', default='save/', help='Directory to save models in.')
parser.add_argument('--data_dir', default='Ambiguous_ReLoc_Dataset/', help='Data directory.')
parser.add_argument('--model', default='model_299', help='Model to restore.')
parser.add_argument('--num_coeff', type=int, default=50, help='Number of components in the mixture model')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size for training.')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--stage', type=int, default=0, help='Define which stage of training to use. 0: Only translation, 1: only rotation, '
                                        'any other number: simultaneously train all distribution components.')
parser.add_argument('--training', action='store_true', default=False, help='Set to True to enable training.')
parser.add_argument('--restore', action='store_true', default=False, help='Set to True to restore the model specified in argument "model".')
parser.add_argument('--base', default='ResNet', help='Specifies the backbone network to use, either "ResNet" (for resnet 34)'
                                                     'or "Inception" (v3).')
parser.add_argument('--prediction_type', default='highest', help='Single best prediction according to the highest mixture coefficient.')

args = parser.parse_args()

# rotation and translation thresholds for evaluation
thresholds = [tuple([10, 0.1]), tuple([15, 0.2]), tuple([20, 0.3]), tuple([60, 1.0])]

# model
model = CamPose(args, device=device, pretrained=True)
crop_size = [224,224]

# image transformations
tforms = [transforms.Resize((256)),
    transforms.RandomCrop(crop_size), transforms.ToTensor()]
data_transform = transforms.Compose(tforms)

# datasets
kwargs = dict(scene=args.scene, data_path=args.data_dir, transform=data_transform)
test_tforms = [transforms.Resize((256)),
          transforms.CenterCrop(crop_size), transforms.ToTensor()]
test_data_transform = transforms.Compose(test_tforms)

kwargs_test = dict(scene=args.scene, data_path=args.data_dir, transform=test_data_transform)
if not args.training:
    kwargs = kwargs_test

if args.dataset == 'ReLoc':
    train_set = AmbiguousRelocData(train=True, **kwargs)
    test_set = AmbiguousRelocData(train=False, **kwargs_test)
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(train_set,
                            batch_size=args.batch_size, shuffle=args.training,
                            num_workers=6, pin_memory=True)

val_loader = torch.utils.data.DataLoader(test_set,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=6, pin_memory=True)

if args.training:
    model.train(train_loader)

print('Train Evaluation')
results = model.eval(train_loader)
hist = eval.compute_pose_histogram(thresholds, results['rotations'], results['translations'], results['labels'])
print(hist)

print('Test Evaluation')
results = model.eval(val_loader)

hist = eval.compute_pose_histogram(thresholds, results['rotations'], results['translations'], results['labels'])
print(hist)
hist = eval.compute_pose_histogram(thresholds, results['rotations_best'], results['translations_best'], results['labels'])
print(hist)

print('Done')


