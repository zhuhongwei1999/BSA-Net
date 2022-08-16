import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from Src.BSANet import BSANet
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import imageio
from skimage import img_as_ubyte

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/final_35.pth')
parser.add_argument('--test_save', type=str,
                    default='./evaluation/Result/')
opt = parser.parse_args()

model = BSANet().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K']:
    save_path = opt.test_save
    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               edge_root='./Dataset/TestDataset/{}/Edge/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # C1 - C4: Pre-refinement layer pred results.
        # R1 - R4: Post-refinement layer pred results.
        # edge: Edge pred results.
        C1, C2, C3, C4, R1, R2, R3, R4, edge = model(image)
        R1 = F.upsample(R1, size=gt.shape, mode='bilinear', align_corners=True)
        R1 = R1.sigmoid().data.cpu().numpy().squeeze()
        R1 = (R1 - R1.min()) / (R1.max() - R1.min() + 1e-8)
        imageio.imsave(save_path + name, img_as_ubyte(R1))
        mae = eval_mae(numpy2tensor(R1), numpy2tensor(gt))
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
