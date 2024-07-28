import os
import sys
import time
import random
import argparse
import warnings

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn

from Pretrained.Loss.CCLoss import cc
from Pretrained.Models.backboneMoCo import IDR, MoCo
from util import adjust_learning_rate, AverageMeter

from Pretrained.DataSet.dataset import GetDataLoader
from Pretrained.DataSet.dataPreprocess import DataPreprocess


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic info
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--save_freq_mc', type=int, default=5, help='save mc frequency')
    parser.add_argument('--save_freq_idr', type=int, default=3, help='save idr frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--ms4_patch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
    # parser.add_argument('--workers', default=32, type=int, help="number of data loading workers (default: 32)")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    # parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--cos', default=True, action="store_true", help='use cosine lr schedule')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # IDRs
    parser.add_argument('--D0', type=int, default=50)
    parser.add_argument('--length', type=int, default=50)
    parser.add_argument('--radius_ratio', type=float, default=0.5)

    # specify folder
    # save IDR 'loss: separate two unsupervised processes', 'resnet not have cbr'
    parser.add_argument('--model_path2', type=str, default='./SaveModel2', help='path to save model')
    # save MoCo 'loss: separate two unsupervised processes', 'resnet not have cbr'
    parser.add_argument('--model_path6', type=str, default='./SaveModel6', help='path to save model')
    # save MoCo 'loss: separate two unsupervised processes', 'resnet not have cbr', 'not use cc loss'
    parser.add_argument('--model_path4', type=str, default='./SaveModel4', help='path to save model')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=1280, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.999, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.07, type=float, help='softmax temperature')

    # other
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    #########################
    opt = parser.parse_args()
    #########################

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

def get_data_loader(args):
    obj = DataPreprocess()
    # _, _, _, _, _, _, ground_xy_all_data, _, _ = obj.data_about()
    _, _, _, _, _, _, _, new_label, new_ground_xy = obj.data_about()
    ms4, pan = obj.get_ms_and_pan()
    get_loader = GetDataLoader(ms4=ms4, pan=pan, patch_size=args.ms4_patch_size, batch_size=args.batch_size)
    return get_loader.get_new_data_loader(new_label=new_label, new_ground_xy=new_ground_xy)
    # return get_loader.get_all_data_loader(ground_xy_allData=ground_xy_all_data)


def set_model_idr(args):
    model_idr = IDR(
        name1='resnet18',
        name='gaussian',
        D0=args.D0,
        radius_ratio=args.radius_ratio,
        length=args.length,
    )
    print(model_idr)

    if torch.cuda.is_available():
        model_idr = model_idr.cuda()
        cudnn.benchmark = True

    return model_idr


def set_model_mc(args):
    model_mc = MoCo(
        name1='resnet18',
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t
    )
    print(model_mc)

    # define the loss function (criterion)
    criterion_fp = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_fm = nn.CrossEntropyLoss().cuda(args.gpu)

    if torch.cuda.is_available():
        model_mc = model_mc.cuda()
        criterion_fp = criterion_fp.cuda()
        criterion_fm = criterion_fm.cuda()
        cudnn.benchmark = True

    return model_mc, criterion_fp, criterion_fm


def set_optimizer_idr(args, model_idr):
    optimizer_idr = torch.optim.SGD(
        model_idr.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    return optimizer_idr


def set_optimizer_mc(args, model_mc):
    optimizer_mc = torch.optim.SGD(
        model_mc.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    return optimizer_mc


def train_idr(data_loader, model_idr, optimizer_idr, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model_idr.train()

    end = time.time()
    for idx, (image_ms, image_pan, _, _) in enumerate(data_loader):
        if idx >= 1150:
            break
        if image_ms.size(0) != args.batch_size:
            continue
        data_time.update(time.time() - end)  # measure data loading time

        bsz = image_ms.size(0)  # bsz = batch size
        image_ms = image_ms.float()
        image_pan = image_pan.float()
        if torch.cuda.is_available():
            image_ms = image_ms.cuda()
            image_pan = image_pan.cuda()

        # ===================forward=====================
        middle1, middle2, middle3, middle4 = model_idr(image_pan, image_ms)
        # CC Loss
        # compute the cc value between j_pan_l and j_ms_l, mutual information
        cc_mutual = cc(middle1, middle2)
        # compute the cc value between j_pan_h and j_ms_h, special information
        cc_special = cc(middle3, middle4)
        loss_cc = (cc_special) ** 2 / (1.01 + cc_mutual)
        # print('cc loss is {}'.format(cc_loss))

        # loss
        loss = 1000 * loss_cc
        # print('loss value is: {}'.format(loss))

        # ===================backward=====================
        # compute gradient and do SGD step
        optimizer_idr.zero_grad()
        loss.backward()
        optimizer_idr.step()

        # ===================meters=======================
        # measure elapsed time
        losses.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                epoch, idx, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def train_mc(data_loader, model_idr, model_mc, criterion_fp, criterion_fm, optimizer_mc, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model_mc.train()

    end = time.time()
    for idx, (image_ms, image_pan, _, _) in enumerate(data_loader):
        if image_ms.size(0) != args.batch_size:
            continue
        data_time.update(time.time() - end)  # measure data loading time

        bsz = image_ms.size(0)  # bsz = batch size
        image_ms = image_ms.float()
        image_pan = image_pan.float()
        if torch.cuda.is_available():
            image_ms = image_ms.cuda()
            image_pan = image_pan.cuda()

        # ===================forward=====================
        # NCE Loss
        # compute output: output1 = logits_fp, output2 = logits_fm, target1 = labels_fp, target2 = labels_fm
        middle1, middle2, middle3, middle4 = model_idr(image_pan, image_ms)
        output1, output2, target1, target2 = model_mc(middle1, middle2, middle3, middle4)

        loss1 = criterion_fp(output1, target1)
        loss2 = criterion_fm(output2, target2)

        # total loss
        loss = 0.5 * loss1 + 0.5 * loss2
        # print('loss value is: {}'.format(loss))

        # ===================backward=====================
        # compute gradient and do SGD step
        optimizer_mc.zero_grad()
        loss.backward()
        optimizer_mc.step()

        # ===================meters=======================
        # measure elapsed time
        losses.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def main1():
    # parse the args
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # set the loader
    all_data_loader, all_data_length = get_data_loader(args=args)
    print('the number of data is {}'.format(all_data_length))

    # set the model
    model_idr = set_model_idr(args=args)

    # set the optimizer
    optimizer_idr = set_optimizer_idr(args=args, model_idr=model_idr)

    args.start_epoch = 1
    cudnn.benchmark = True

    # training
    for epoch in range(args.start_epoch, args.epochs+1):

        adjust_learning_rate(epoch, args, optimizer_idr)

        print("==> training...")
        time1 = time.time()

        train_idr(data_loader=all_data_loader,
                  model_idr=model_idr,
                  optimizer_idr=optimizer_idr,
                  epoch=epoch,
                  args=args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq_idr == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model_idr.state_dict(),
                'optimizer': optimizer_idr.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_path2, 'ckpt_epoch_{epoch}_bei.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


def main2():
    # parse the args
    args = parse_option()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # set the loader
    all_data_loader, all_data_length = get_data_loader(args=args)

    # set the model_idr
    model_idr = set_model_idr(args=args)
    checkpoint = torch.load('../Partone/SaveModel2/ckpt_epoch_xx_bei.pth', map_location='cpu')
    model_weights = checkpoint['model']
    model_idr.load_state_dict(model_weights)

    # set the model_mc
    model_mc, criterion_fp, criterion_fm = set_model_mc(args=args)

    # set the optimizer
    optimizer_mc = set_optimizer_mc(args=args, model_mc=model_mc)

    args.start_epoch = 1
    cudnn.benchmark = True

    # training
    for epoch in range(args.start_epoch, args.epochs+1):

        adjust_learning_rate(epoch, args, optimizer_mc)

        print("==> training...")
        time1 = time.time()
        train_mc(data_loader=all_data_loader,
                 model_idr=model_idr,
                 model_mc=model_mc,
                 criterion_fp=criterion_fp,
                 criterion_fm=criterion_fm,
                 optimizer_mc=optimizer_mc,
                 epoch=epoch,
                 args=args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq_mc == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model_mc.state_dict(),
                'optimizer': optimizer_mc.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_path6, 'ckpt_epoch_{epoch}_bei.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # main1()
    main2()
