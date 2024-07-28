# import sys, os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
import os
import cv2
import torch
import argparse
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
# from ..Pretrained.util import adjust_learning_rate

from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from classification_net import IDR, FCN, MoCo
from Pretrained.DataSet.dataset import GetDataLoader
from Pretrained.DataSet.dataPreprocess import DataPreprocess


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic info
    parser.add_argument('--epoch', type=int, default=66, help='train epochs')
    parser.add_argument('--dim', type=int, default=10, help='category numbers')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--save_freq', type=int, default=3, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--ms4_patch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    # parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--cos', default=True, action="store_true", help='use cosine lr schedule')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # IDRs
    parser.add_argument('--D0', type=int, default=50)
    parser.add_argument('--length', type=int, default=50)
    parser.add_argument('--radius_ratio', type=float, default=0.5)

    # specify folder
    parser.add_argument('--model_path_dt1', type=str, default='./SaveModelDT1',help='path to save model')
    parser.add_argument('--model_path_dt2', type=str, default='./SaveModelDT2', help='path to save model')
    parser.add_argument('--model_path_dt3', type=str, default='./SaveModelDT3', help='path to save model')
    parser.add_argument('--model_path_dt4', type=str, default='./SaveModelDT4', help='path to save model')
    parser.add_argument('--model_path_dt5', type=str, default='./SaveModelDT5', help='path to save model')

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


def get_all_data_loader(args):
    obj = DataPreprocess(ms4_url='../Partone/Data9/Image9/ms4.tif',
                         pan_url='../Partone/Data9/Image9/pan.tif',
                         label_np_url='../Partone/Data9/Image9/label.npy')
    _, _, all_label, _, _, all_ground_xy, ground_xy_all_data, _, _ = obj.data_about()
    ms4, pan = obj.get_ms_and_pan()
    print('MS size is: {}'.format(ms4.size()))
    print('PAN size is: {}'.format(pan.size()))
    # 获取 Data loader, ms4_patch_size=16
    get_loader = GetDataLoader(ms4=ms4, pan=pan, patch_size=args.ms4_patch_size, batch_size=args.batch_size)

    # data_loader, n_data
    return get_loader.get_all_data_loader(ground_xy_allData=ground_xy_all_data)  # image_ms, image_pan, locate_xy, index
    # return get_loader.get_all_mark_data_loader(all_label=all_label, all_ground_xy=all_ground_xy)  # image_ms, image_pan, target, locate_xy, index



def set_model(args):
    model1 = IDR(
        name1='resnet18',
        name='gaussian',
        D0=args.D0,
        radius_ratio=args.radius_ratio,
        length=args.length
    )
    model2 = MoCo(name1='resnet18')
    fc = FCN(args.dim)
    print(model1)
    print(model2)
    print(fc)

    if torch.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
        fc = fc.cuda()
        cudnn.benchmark = True

    return model1, model2, fc


def color(model1, model2, fc, dataloader):
    color = np.array([[255, 255, 0], [255, 0, 0], [33, 145, 237], [201, 252, 189], [0, 0, 230], [0, 255, 0],
                      [240, 32, 160], [221, 160, 221], [140, 230, 240], [0, 255, 255]])
    class_count = np.zeros(10)
    out_color = np.zeros((6905, 7300, 3))
    model1.eval()
    model2.eval()
    fc.eval()
    with torch.no_grad():
        for step, (ms, pan, gt_xy, index) in enumerate(tqdm(dataloader)):
        # for step, (ms, pan, _, gt_xy, index) in enumerate(tqdm(dataloader)):
            # print(gt_xy.size())
            # print(index)
            ms, pan = ms.cuda(), pan.cuda()
            j_pan_l, j_ms_l, j_pan_h, j_ms_h = model1(pan, ms)
            fusion_feature = model2(j_pan_l, j_ms_l, j_pan_h, j_ms_h)
            output = fc(fusion_feature)
            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
            pred_y_numpy = pred_y.cpu().numpy()
            gt_xy = gt_xy.numpy()
            for k in range(len(gt_xy)):
                # print(len(gt_xy))
                class_count[pred_y_numpy[k]] = class_count[pred_y_numpy[k]] + 1
                out_color[gt_xy[k][0]][gt_xy[k][1]] = color[pred_y_numpy[k]]
        # cv2.imwrite("Beijing1.png", out_color)
        cv2.imwrite("Beijing.png", out_color)



def main():
    # set arguments
    args = parse_option()

    # set the model, model1: idr; model2: MoCo
    model1, model2, fc = set_model(args=args)

    # compute the number of parameters
    total_params1 = sum(p.numel() for p in model1.CBRpl1.parameters())
    total_params2 = sum(p.numel() for p in model1.EncoderPH.layer1.parameters())
    total_params3 = sum(p.numel() for p in model2.FM_low.parameters())
    print(f"the number of parameters: {total_params1 * 8 + total_params2 * 10 + total_params3 * 3}")

    # load weights dict, use contrastive loss and cc loss
    checkpoint = torch.load('../DownstreamOne/SaveModelDT1/ckpt_epoch_xx_bei_l2.pth', map_location='cpu')

    weights_1 = checkpoint['model1']
    weights_2 = checkpoint['model2']
    weights_fc = checkpoint['fc']

    # load weights dict to model
    model1.load_state_dict(weights_1)
    model2.load_state_dict(weights_2)
    fc.load_state_dict(weights_fc)

    # get loader
    data_loader, data_length = get_all_data_loader(args=args)
    # color
    color(model1, model2, fc, data_loader)


if __name__ == '__main__':
    main()