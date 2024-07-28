import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Pretrained.util import adjust_learning_rate

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
    parser.add_argument('--model_path_dt', type=str, default='./SaveModelDT', help='path to save model')
    parser.add_argument('--model_path_dt1', type=str, default='./SaveModelDT1',help='path to save model')
    parser.add_argument('--model_path_dt2', type=str, default='./SaveModelDT2', help='path to save model')

    parser.add_argument('--model_path', type=str, default='./SaveModel', help='path to save model')

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


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(0, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    return aa, oa, on_display


def get_train_data_loader(args):
    obj = DataPreprocess(ms4_url='../Partone/Data9/Image9/ms4.tif',
                         pan_url='../Partone/Data9/Image9/pan.tif',
                         label_np_url='../Partone/Data9/Image9/label.npy')
    label_train, _, _, ground_xy_train, _, _, _, _, _ = obj.data_about()
    ms4, pan = obj.get_ms_and_pan()
    get_loader = GetDataLoader(ms4=ms4, pan=pan, patch_size=args.ms4_patch_size, batch_size=args.batch_size)
    # data_loader, n_data
    return get_loader.get_train_data_loader(label_train=label_train, ground_xy_train=ground_xy_train)


def get_test_data_loader(args):
    obj = DataPreprocess(ms4_url='../Partone/Data9/Image9/ms4.tif',
                         pan_url='../Partone/Data9/Image9/pan.tif',
                         label_np_url="../Partone/Data9/Image9/label.npy")
    _, label_test, _, _, ground_xy_test, _, _, _, _ = obj.data_about()
    ms4, pan = obj.get_ms_and_pan()
    get_loader = GetDataLoader(ms4=ms4, pan=pan, patch_size=args.ms4_patch_size, batch_size=args.batch_size)
    # data_loader, n_data
    return get_loader.get_test_data_loader(label_test=label_test, ground_xy_test=ground_xy_test)


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


def set_scheduler(args, model1, model2, fc):
    optimizer = torch.optim.Adam([{'params': model1.parameters()},
                                  {'params': model2.parameters(), 'lr': args.learning_rate},
                                  {'params': fc.parameters(), 'lr': args.learning_rate}],
                                 lr=args.learning_rate,
                                 weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, 15, gamma=0.1, last_epoch=-1)
    return optimizer, scheduler


def set_optimizer(args, model1, model2, fc):
    optimizer = torch.optim.SGD(
        [{'params': model1.parameters()},
         {'params': model2.parameters(), 'lr': args.learning_rate},
         {'params': fc.parameters(), 'lr': args.learning_rate}],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer


def train_model(model1, model2, fc, train_loader, optimizer, scheduler, epoch):
    # update lr
    if epoch == 15:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    elif epoch == 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    model1.train()
    model2.train()
    fc.train()
    correct = 0.0
    for step, (ms, pan, label, _, _) in enumerate(train_loader):
        # print('ms shape is {}'.format(ms.shape))
        ms, pan, label = ms.cuda(), pan.cuda(), label.cuda()

        optimizer.zero_grad()

        j_pan_l, j_ms_l, j_pan_h, j_ms_h = model1(pan, ms)
        fusion_feature = model2(j_pan_l, j_ms_l, j_pan_h, j_ms_h)
        output = fc(fusion_feature)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())

        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    train_acc = correct * 100.0 / len(train_loader.dataset)
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))
    print("the {}_th epoch's learning rate is: {}".format(epoch, optimizer.param_groups[0]['lr']))
    # Update learning rate
    # scheduler.step()

    return train_acc


def ceshi(model1, model2, fc, test_loader, categories_number):
    model1.eval()
    model2.eval()
    fc.eval()
    correct = 0.0
    test_metric = np.zeros([categories_number, categories_number])

    with torch.no_grad():
        for step, (ms, pan, target, _, _) in enumerate(tqdm(test_loader)):
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            j_pan_l, j_ms_l, j_pan_h, j_ms_h = model1(pan, ms)
            fusion_feature = model2(j_pan_l, j_ms_l, j_pan_h, j_ms_h)
            output = fc(fusion_feature)
            # test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]

            for i in range(len(target)):
                test_metric[int(pred[i].item())][int(target[i].item())] += 1
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_accuracy = (100.0 * correct) / len(test_loader.dataset)
        print("test accuracy is:{:.3f} \n".format(test_accuracy))

    average_accuracy, overall_accuracy, _ = aa_oa(test_metric)
    kappa_coefficient = kappa(test_metric)
    print('AA: {0:f}'.format(average_accuracy))
    print('OA: {0:f}'.format(overall_accuracy))
    print('KAPPA: {0:f}'.format(kappa_coefficient))
    return test_accuracy


def main():
    # set arguments
    args = parse_option()
    categories_number = args.dim

    # set the model
    model1, model2, fc = set_model(args=args)

    # compute the number of parameters
    total_params1 = sum(p.numel() for p in model1.CBRpl1.parameters())
    total_params2 = sum(p.numel() for p in model1.EncoderPH.layer1.parameters())
    total_params3 = sum(p.numel() for p in model2.FM_low.parameters())
    print(f"the number of parameters: {total_params1 * 8 + total_params2 * 10 + total_params3 * 3}")

    # # load weights dict
    # checkpoint_1 = torch.load('../Partone/SaveModel2/ckpt_epoch_xx_bei.pth', map_location='cpu')
    # checkpoint_2= torch.load('../Partone/SaveModel6/ckpt_epoch_xx_bei.pth', map_location='cpu')
    # weights_1 = checkpoint_1['model']
    # weights_2 = checkpoint_2['model']

    # load weights dict, last experiment
    checkpoint_1 = torch.load('../Partone/SaveModelCC/ckpt_epoch_xx_bei.pth', map_location='cpu')
    checkpoint_2 = torch.load('../Partone/SaveModelCL/ckpt_epoch_xx_bei.pth', map_location='cpu')
    weights_1 = checkpoint_1['model']
    weights_2 = checkpoint_2['model']

    # delete some keys
    del_keys = ["queue_pan", "queue_ms", "queue_ptr", "FM_pan.CBR.conv1.weight", "FM_pan.CBR.bn1.weight", "FM_pan.CBR.bn1.bias", "FM_pan.CBR.bn1.running_mean", "FM_pan.CBR.bn1.running_var", "FM_pan.CBR.bn1.num_batches_tracked", "FM_ms.CBR.conv1.weight", "FM_ms.CBR.bn1.weight", "FM_ms.CBR.bn1.bias", "FM_ms.CBR.bn1.running_mean", "FM_ms.CBR.bn1.running_var", "FM_ms.CBR.bn1.num_batches_tracked"]
    for k in del_keys:
        del weights_2[k]

    # load weights dict to model
    model1.load_state_dict(weights_1)
    model2.load_state_dict(weights_2)

    # set the optimizer
    # optimizer = set_optimizer(args=args, model=model, fc=fc)
    optimizer, scheduler = set_scheduler(args=args, model1=model1, model2=model2, fc=fc)

    # get loader(train and test)
    train_data_loader, train_data_length = get_train_data_loader(args=args)
    test_data_loader, test_data_length = get_test_data_loader(args=args)

    for epoch in range(1, args.epoch + 1):
        # adjust_learning_rate(epoch, args, optimizer)  # match the set_optimizer function
        # train
        print("==> training...")
        train_acc = train_model(model1, model2, fc, train_data_loader, optimizer, scheduler, epoch)

        # save
        if epoch >= 16 and epoch % 2 == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model1': model1.state_dict(),
                'model2': model2.state_dict(),
                'fc': fc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_path, 'ckpt_epoch_{epoch}_bei.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        # test
        if epoch >= 16 and epoch % 2 == 0:
            print("==> testing...")
            test_acc = ceshi(model1, model2, fc, test_data_loader, categories_number)


def main1():
    # set arguments
    args = parse_option()
    categories_number = args.dim

    # set the model
    model1, model2, fc = set_model(args=args)

    # load weights dict
    checkpoint = torch.load('../DownstreamOne/SaveModelDT1/ckpt_epoch_xx_bei_r2.pth', map_location='cpu')
    weights_1 = checkpoint['model1']
    weights_2 = checkpoint['model2']
    weights_3 = checkpoint['fc']

    # load weights dict to model
    model1.load_state_dict(weights_1)
    model2.load_state_dict(weights_2)
    fc.load_state_dict(weights_3)

    # get loader(train and test)
    test_data_loader, test_data_length = get_test_data_loader(args=args)
    print("==> testing...")
    test_acc = ceshi(model1, model2, fc, test_data_loader, categories_number)

    print('test acc is {}'.format(test_acc))


if __name__ == '__main__':
    main()  # train and test
    # main1()  # only test
