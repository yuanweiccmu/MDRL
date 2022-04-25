import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from data_list import ImageList
from utils import accuracy, AverageMeter, save_checkpoint, save_stu_checkpoint
from torch.utils.data import DataLoader
import os.path
import pandas as pd
import numpy as np
import roi_net
from requests.utils import urlparse
import wget
import logging
import datetime
import matplotlib as mpl
mpl.rcParams.update({
    # 'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman'],
    })
import xlrd

# import torchvision.datasets as dataset
# Data loading code
train_bs = 64
# test_bs = 16
fine_bs = 32
test_num = 129
# lmda = 0.3
model_path = 'model'
test_part_path = 'data/oct'
train_txt_path = open(os.path.join("data/oct", "train_part.txt")).readlines()
test_txt_path = open(os.path.join("data/oct", "test_part.txt")).readlines()
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
normTransform_v = transforms.Normalize([0.5], [0.5])
# device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

trainTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.repeat(3,1,1)),
    normTransform
])
testTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normTransform
])

train_data = ImageList(image_list=train_txt_path, args='other', transform=trainTransform)
test_data = ImageList(image_list=test_txt_path, args='other', transform=testTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=train_bs, shuffle=False)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--evaluate-freq', default=10, type=int,
                    help='the evaluation frequence')
parser.add_argument('--resume', default='./checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--stu_resume', default='./stu_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest stu_checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--n_classes', default=2, type=int,
                    help='the number of classes')
parser.add_argument('--n_samples', default=2, type=int,
                    help='the number of samples per class')
parser.add_argument('--net', default='resnet50', type=str,
                    help='baseline')
parser.add_argument('-heat_dir', default='heatmap/raw/', type=str, help='heatmap_dir')
parser.add_argument('-save_heat_dir', default='heatmap/', type=str, help='where to save heatmap')

best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(lmda):
    global args, best_prec1
    args = parser.parse_args()
    args.net = 'roinet'
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    print('load {}'.format(args.net))
    # create model
    if args.net[0:3] == 'roi':
        model = roi_net.resnet50(args.n_classes, lmda)
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        model_dir = 'model'
        filename = os.path.basename(urlparse(url).path)
        pretrained_path = os.path.join(model_dir, filename)
        if not os.path.exists(pretrained_path):
            wget.download(url, pretrained_path)
        if pretrained_path:
            logging.info('load pretrained backbone')
            net_dict = model.state_dict()
            pretrained_dict = torch.load(pretrained_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            model.load_state_dict(net_dict)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print('loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('no checkpoint found at {}'.format(args.resume))

    cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 * len(train_loader))
    step = 0
    print('START TIME:', time.asctime(time.localtime(time.time())))
    t = datetime.datetime.now()
    with open(os.path.join(model_path, 'train_time.txt'), 'a') as f:
        f.write('the start time is: {}'.format(t))
        f.write('\n')
    for epoch in range(args.start_epoch, args.epochs):
        if args.net[0:3] == 'roi':
            step, Loss, Acc, Acc_part = train_roi(train_loader, model, criterion, optimizer, scheduler, epoch,
                         step, 'part_acc')


def train_roi(train_loader, model, criterion, optimizer, scheduler, epoch, step, mode=None):
    global best_prec1

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = 0
    correct = 0
    total = 0
    Loss = []
    Acc = []
    Acc_part = []
    # switch to train mode
    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = input.to(device)
        target_var = target.to(device).squeeze()
        # print(label)

        # compute output
        loss_ret, acc_ret, mask_cat, logits, _ = model(input_var, target_var, 'train')

        # compute loss
        softmax_loss = loss_ret['loss']
        loss = softmax_loss

        # measure acccuracy and record loss
        prec1 = accuracy(logits, target_var, 1)
        train_loss += loss.data
        total += target_var.size(0)
        correct += acc_ret['acc']

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if epoch >= 0:
            optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        losses = train_loss / (i + 1)
        if i % args.print_freq == 0:
            print(' Step: {} total: {} Epoch: {} Loss :{} ({}) Prec@1 {} Prec@2 {}'.format(i, len(train_loader),
                                                                                 epoch, loss, train_loss / (i + 1),
                                                                                 100. * correct / total, prec1))
        if i == len(train_loader) - 1:
            if mode == 'part_acc':
                prec1, part_acc = validate_roi(test_loader, model, criterion, mode)
                is_best = part_acc > best_prec1
                best_prec1 = max(part_acc, best_prec1)
                print('the best acc is :', prec1)
                print('the part acc is :', part_acc)
                with open(os.path.join(model_path, 'acc.txt'), 'a') as f:
                    f.write('best acc: ' + str(prec1) + '   part acc: ' + str(part_acc))
                    f.write('\n')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            Acc.append(prec1)
            Acc_part.append(part_acc)
            Loss.append(losses)

        step = step + 1
    return step, Loss, Acc, Acc_part


def validate_roi(test_loader, model, criterion, mode):
    test_loss = 0
    correct = 0
    total = 0
    P = []
    part_acc = 0
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(test_loader):
            input_var = input.to(device)
            target_var = target.to(device).squeeze()

            # compute output
            loss_ret, acc_ret, mask_cat, logits, _ = model(input_var, target_var, 'test')
            softmax_loss = loss_ret['loss']
            loss = softmax_loss
            _, pre = logits.topk(1, 1, True, True)
            for j in pre:
                P.append(j.data.cpu().numpy().astype(int)[0])
            prec1 = accuracy(logits, target_var, 1)
            test_loss += loss.data
            total += target_var.size(0)
            correct += acc_ret['acc']

            if i % args.print_freq == 0:
                print(' Step: {} total: {} Loss :{} Prec@1 {} Prec@2 {}'.format(i, len(test_loader),
                                                                      loss, 100. * correct / total, prec1))
        # print(type(P[0].tolist()))
        if mode == 'part_acc':
            with open(os.path.join(model_path, 'pre.txt'), 'w') as f:
                for i in range(len(P)):
                    if P[i] == 0:
                        f.write('0')
                        f.write('\n')
                    if P[i] == 1:
                        f.write('1')
                        f.write('\n')
            part_acc, _, _, _ = test_part(pre_txt_path=model_path, test_txt_path = test_part_path, test_num=test_num)

    return 100. * correct / total, part_acc


def test(test_loader, lmda, mode=None):
    global args, best_prec1
    args = parser.parse_args()
    args.net = 'roinet'
    print('load {}'.format(args.net))
    # create model
    if args.net[0:3] == 'roi':
        model = roi_net.resnet50(args.n_classes, lmda)
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        model_dir = 'model'
        filename = os.path.basename(urlparse(url).path)
        pretrained_path = os.path.join(model_dir, filename)
        if not os.path.exists(pretrained_path):
            wget.download(url, pretrained_path)
        if pretrained_path:
            logging.info('load pretrained backbone')
            net_dict = model.state_dict()
            pretrained_dict = torch.load(pretrained_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            model.load_state_dict(net_dict)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)

    if os.path.isfile('model_best.pth.tar'):
        print('loading checkpoint {}'.format('model_best.pth.tar'))
        checkpoint = torch.load('model_best.pth.tar')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded checkpoint {}(epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('no checkpoint found at {}'.format(args.resume))
    if mode == 'part_acc':
        prec1, part_acc = validate_roi(test_loader, model, criterion, mode)
        print('the best acc is :', prec1)
        print('the part acc is :', part_acc)
        with open(os.path.join(model_path, 'test_acc.txt'), 'a') as f:
            f.write('best acc: ' + str(prec1) + '   part acc: ' + str(part_acc))
            f.write('\n')
    else :
        prec1, _ = validate_roi(test_loader, model, criterion, mode)
        print('the best acc is :', prec1)


def test_part(pre_txt_path, test_txt_path, test_num):
    df = pd.read_excel('label.xlsx', sheet_name='Sheet1', usecols=[2, 5, 6])
    t = []
    he = []
    normal = []
    abnormal = []
    final = []
    for i in range(test_num):
        normal.append(0)
        abnormal.append(0)
        final.append(0)
    # print()
    for i in range(len(df['file_name'])):
        if i == 0:
            temp = 1
            df['file_name'][i] = temp
        #         print(temp)
        else:
            if pd.isnull(df['file_name'][i]) == True:
                df['file_name'][i] = temp
            #             print(df['file_name'][i])
            else:
                temp = temp + 1
                df['file_name'][i] = temp
    # print(df.head(20))
    for i in range(len(df['file_name'])):
        if i == 0:
            flag = 1
            t.append(df['oct_label'][i])
            he.append(df['he_label'][i])
        else:
            if df['file_name'][i] == flag + 1:
                t.append(df['oct_label'][i])
                he.append(df['he_label'][i])
                flag = flag + 1
                # if pd.isnull(df['oct_label'][i]) == True:
                #     # print(df['file_name'][i])
            else:
                continue
    # print(t)
    lst = open(os.path.join(test_txt_path, "test_part.txt")).readlines()
    pre = open(os.path.join(pre_txt_path, "pre.txt")).readlines()
    # print(lst[0].split()[0].split("\\")[-1].split('_')[0])
    # print(pre[0][0])
    for i in range(len(lst)):
        inx = int(lst[i].split()[0].split("/")[-1].split('_')[0]) - 1
        if pre[i][0] == '0':
            normal[inx] = normal[inx] + 1
        if pre[i][0] == '1':
            abnormal[inx] = abnormal[inx] + 1
    # print(normal)
    # print(abnormal)
    for i in range(test_num):
        if normal[i] == abnormal[i] == 0:
            continue
        elif normal[i] == abnormal[i] != 0:
            final[i] = '+'
        elif normal[i] > abnormal[i]:
            final[i] = '-'
        elif normal[i] < abnormal[i]:
            final[i] = '+'
    # print(final)

    n = 0
    c = 0
    acc = 0
    y_true = []
    y_predict = []
    he_true = []
    for i in range(test_num):
        if final[i] == 0:
            continue
        else:
            if final[i] == '+':
                y_predict.append(1)
            else:
                y_predict.append(0)
            if t[i] == '+':
                y_true.append(1)
            else:
                y_true.append(0)
            if he[i] == '+':
                he_true.append(1)
            else:
                he_true.append(0)
            if final[i] == t[i]:
                n = n + 1
                c = c + 1
            else:
                n = n + 1
    acc = c / n * 100
    y_true = torch.tensor(np.array(y_true))
    y_predict = torch.tensor(np.array(y_predict))
    he_true = torch.tensor(np.array(he_true))
    return acc, y_true, y_predict, he_true


if __name__ == '__main__':
    train_step = False
    if train_step:
        main(lmda=0.5)
    else:
        test(test_loader=test_loader, lmda=0.5, mode='part_acc')









