"""
   point cloud pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Haowen Deng
"""

import argparse
import sys

import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import loss_functions
import network_bingham
from data_modelnet import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 2048]')
parser.add_argument('--num_model', type=int, default=50, help='Number of models used')
parser.add_argument('--max_epoch', type=int, default=50000, help='Epoch to run [default: 50000]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--cls_id', type=int, default=0, help='The id of the target object class')
parser.add_argument('--weight_fn', default=None, help='Pre-trained weights for the network')

m10_clsses = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
              'night_stand', 'sofa', 'table', 'toilet']

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(flags.gpu)
cls = m10_clsses[flags.cls_id]
num_point = flags.num_point
nm = flags.num_model
max_epoch = flags.max_epoch
batch_size = flags.batch_size
learning_rate = flags.learning_rate

if nm > 1:
    print('Training MBN with {} branches'.format(nm))
elif nm == 1:
    print('Training UBN')
else:
    print('Invalid number of branches. Exit')
    sys.exit(-1)

save_dir = flags.log_dir + '_{}'.format(cls)
print('save dir: {}'.format(save_dir))

device = torch.device('cuda:0')
weight_fn = None
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

log_fn = os.path.join(save_dir, 'log.txt')
fid = open(log_fn, 'w')
summary_writer = SummaryWriter(os.path.join(save_dir, 'summary'))

## default loss is rwta + mb loss
criterion = loss_functions.RWTA_MB_Loss(nm)


## Comment out the next line to use rwta + ce loss
# criterion = loss_functions.RWTA_CE_Loss(nm)


def run_one_epoch(net, dataloader, optimizer, epoch, is_training=True):
    if is_training:
        net.train()
    else:
        net.eval()
    rwta_loss_lst = []
    mb_loss_lst = []
    loss_lst = []

    n_batch = len(dataloader)
    for batch_idx, (data, label) in enumerate(dataloader):
        if is_training:
            net.zero_grad()

        points = data.float().to(device)
        label = label.float().to(device)
        label = ((label[:, 0:1] > 0).float() - 0.5) * 2 * label

        pred_q, pred_l, weights = net(points)
        rwta_loss, mb_loss = criterion(pred_q, pred_l, weights, label)

        loss = 1.0 * rwta_loss + mb_loss

        rwta_loss_lst.append(rwta_loss.item())
        mb_loss_lst.append(mb_loss.item())
        loss_lst.append(loss.item())

        if is_training:
            loss.backward()
            optimizer.step()

        print("{}/{}".format(batch_idx, n_batch), end='\r')

    mean_rwta_loss = np.mean(rwta_loss_lst)
    mean_mb_loss = np.mean(mb_loss_lst)
    mean_loss = np.mean(loss_lst)

    print_str = "Epoch: {} Loss:{:.4f} RWTA:{:.4f} MB:{:.4f}".format(epoch, mean_loss, mean_rwta_loss, mean_mb_loss)
    if is_training:
        print(print_str)
        fid.write(print_str + '\n')
        summary_writer.add_scalar('train/loss', mean_loss, epoch)
        summary_writer.add_scalar('train/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('train/mb_loss', mean_mb_loss, epoch)
    else:
        print('Eval ' + print_str)
        fid.write('Eval ' + print_str + '\n')
        summary_writer.add_scalar('eval/loss', mean_loss, epoch)
        summary_writer.add_scalar('eval/rwta_loss', mean_rwta_loss, epoch)
        summary_writer.add_scalar('eval/mb_loss', mean_mb_loss, epoch)

    return mean_loss


def main():
    from torch.utils.data import DataLoader
    from torchvision import transforms
    data = load_modelnet10(cls, 'train')[:, :, :3]
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.2)
    print('train_data: {} val_data: {}'.format(train_data.shape, val_data.shape))

    m_transforms = transforms.Compose([
        PointDownsample(num_point)
    ])

    train_ds = ModelNetDatasetIcoshpere(train_data, m_transforms)
    val_ds = ModelNetDatasetIcoshpere(val_data, m_transforms)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    net = network_bingham.MBN(num_point, 3, 128, nm)
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)
    if weight_fn is not None:
        print('Loading from {}'.format(weight_fn))
        state_dict = torch.load(weight_fn)
        net.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            run_one_epoch(net, val_dl, None, -1, is_training=False)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(max_epoch):
        run_one_epoch(net, train_dl, optimizer, epoch, is_training=True)

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_loss = run_one_epoch(net, val_dl, None, epoch, is_training=False)
                lr_scheduler.step(val_loss)

    ## save weights
    save_state_fn = os.path.join(save_dir, 'chkpt.pth')
    torch.save(net.state_dict(), save_state_fn)

    ## close file handler
    fid.close()

    print('Training for {} is finished'.format(cls))


if __name__ == '__main__':
    main()
