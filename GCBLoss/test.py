import torch
import config as cfg
from lib.test_loader import TestDataSet
from tqdm import tqdm
import torch.nn.functional as F
from lib.utils import calc_metric
import cv2
import os
import shutil
import numpy as np
import pdb

# 指定模型路径
weight_path = 'weights/CVC-ClinicDB_AHENet/BestNet.pth'
from lib.network_level_4 import Network

vis_dir = './vis/'+weight_path.split('/')[-2]+'/'
print(vis_dir)
if os.path.exists(vis_dir):
    shutil.rmtree(vis_dir)
os.makedirs(vis_dir)

 
test_dataset = TestDataSet(cfg)

net = Network(base_channel=cfg.base_channel, inter_channel=cfg.inter_channel, rdb_block_num=cfg.rdb_block_num)
print('Weight Path:', weight_path)
net.load_state_dict(torch.load(weight_path))
net.cuda()


torch.set_grad_enabled(False)
net.eval()
dice_score_all = None
f1_score_all = None
iou_score_all = None

for img_batch, label_batch, img_path_batch in tqdm(test_dataset.loader, total=test_dataset.loader.__len__(), ncols=0):
    b = img_batch.size(0)
    img_batch = img_batch.cuda()        # (b, 3, h, w)
    label_batch = label_batch.cuda()    # (b, 1, h, w)
    #print(img_batch.shape)
    pre_batch = net(img_batch)          # (b, 1, h, w)
    pre_batch = F.interpolate(pre_batch, size=(label_batch.size(2), label_batch.size(3)))
    pre_batch = torch.sigmoid(pre_batch)
    pre_batch[pre_batch > 0.5] = 1
    pre_batch[pre_batch <= 0.5] = 0
    f1_score, iou_score = calc_metric(pre_batch.long(), label_batch.long())
    
    if iou_score_all is None:
        #dice_score_all = dice
        f1_score_all = f1_score
        iou_score_all = iou_score
    else:
        #dice_score_all = torch.cat([dice_score_all, dice], 0) 
        f1_score_all = torch.cat([f1_score_all, f1_score], 0)
        iou_score_all = torch.cat([iou_score_all, iou_score], 0)

    for index in range(pre_batch.size(0)):
        pre = pre_batch[index][0]
        label = label_batch[index][0]

        img_path = img_path_batch[index]
        img = cv2.imread(img_path)

        pre = pre.cpu().long().numpy() * 255
        label = label.cpu().long().numpy() * 255
        #print(img.shape, pre.shape, label.shape)
        img = np.concatenate([img, pre[:, :, None].repeat(3, 2), 255*np.ones((img.shape[0], 1, 3)), label[:, :, None].repeat(3, 2)], 1)

        img_name = '%s.png' % (img_path.split('/')[-1].split('.')[0])
        print('%s F1: %.4f' % (img_path.split('/')[-1].split('.')[0], f1_score))
        vis_path = os.path.join(vis_dir, img_name)
        cv2.imwrite(vis_path, img)


print('[Average_F1 %.4f Average_IOU %.4f]' % ( f1_score_all.mean(), iou_score_all.mean()))

