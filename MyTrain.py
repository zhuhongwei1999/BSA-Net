import torch
import argparse
from Src.BSANet import BSANet
from Src.utils.Dataloader import get_loader
from Src.utils.trainer import trainer, adjust_lr
from apex import amp
import torch.nn as nn
import torch.nn.functional as F

def structure_loss(pred, mask):

    # BCE loss
    k = nn.Softmax2d()
    weit = torch.abs(pred-mask)
    weit = k(weit)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # IOU loss
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1-(inter+1)/(union-inter+1)

    return (wbce + wiou).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=35,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=36,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=384,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Imgs/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')
    parser.add_argument('--train_edge_dir', type=str, default='./Dataset/TrainDataset/Edge/')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    model_BSANet = BSANet().cuda()

    optimizer = torch.optim.Adam(model_BSANet.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()
    net, optimizer = amp.initialize(model_BSANet, optimizer, opt_level='O0')
    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, opt.train_edge_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)
    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)
    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        trainer(train_loader=train_loader, model=model_BSANet,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=structure_loss, total_step=total_step)
