import torch
import config as cfg
import timeit
from  lib.train_loader import TrainDataSet
from lib.test_loader import TestDataSet
import torch.nn.functional as F
from lib.utils import calc_metric
import os
import shutil

from loss import WeightedDiceBCE
from tensorboardX import SummaryWriter

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = 10 * lr
    return lr


assert not os.path.exists(cfg.weight_dir), 'please run [ rm -r %s ]' % cfg.weight_dir
os.makedirs(cfg.weight_dir)


lib_path = './lib/'
assert os.path.exists(lib_path), lib_path
shutil.copytree(lib_path, os.path.join(cfg.weight_dir, 'lib/'))

config_path = './config.py'
assert os.path.exists(config_path), config_path
shutil.copy(config_path, os.path.join(cfg.weight_dir, 'config.py'))


from lib.network_level_4 import Network



train_dataset = TrainDataSet(cfg)
test_dataset = TestDataSet(cfg)
net = Network(
    base_channel=cfg.base_channel, 
    inter_channel=cfg.inter_channel, 
    rdb_block_num=cfg.rdb_block_num,
)
net.cuda()

opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-5)
if cfg.reload_from_checkpoint:
    print('loading from checkpoint: {}'.format(cfg.reload_path))
    if os.path.exists(cfg.reload_path):
        net.load_state_dict(torch.load(cfg.reload_path, map_location=torch.device('cpu')))
        torch.save(net.state_dict(), os.path.join(cfg.weight_dir, 'NewestNet_copy.pth'))
    else:
        print('File not exists in the reload path: {}'.format(cfg.reload_path))


loss_func = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)

BEST_EPOCH = 0
BEST_F1 = 0
BEST_IOU = 0
start = timeit.default_timer()
writer = SummaryWriter(cfg.weight_dir+'/')
for epoch_index in range(cfg.epoch):
    # if epoch < cfg.start_epoch:
    #     continue
    # ------------------------------------- train ----------------------------------------------
    total_loss = 0
    total_sample = 0
    torch.set_grad_enabled(True)
    net.train()
    if cfg.lr_decay:
        if epoch_index < 3000:
            adjust_learning_rate(opt, epoch_index, cfg.lr, cfg.epoch, power=0.9)

    for img_batch, label_batch in train_dataset.loader:
        #print('img_batch.shape:', img_batch.shape)
        b = img_batch.size(0)
        img_batch = img_batch.cuda()        # (b, 3, h, w)
        label_batch = label_batch.cuda()    # (b, 1, h, w)
        pre_batch, loss_list = net(img_batch, label_batch)
        loss = loss_func(pre_batch, label_batch) + loss_list[0].mean() #+ loss_list[1].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss * b
        total_sample += b
    
    # opt.param_groups[0]['lr'] = lr_origin * (1 - epoch_index / cfg.epoch)
    mean_loss = total_loss / total_sample
    # print('[Epoch %d] [loss %.4f]' % (epoch_index, mean_loss.item(),))

    # ----------------------------------------- test -----------------------------------------------
    if epoch_index == 0 or (epoch_index+1) % cfg.test_per_ep == 0:
        f1_score_all = None
        iou_score_all = None
        torch.set_grad_enabled(False)
        net.eval()
        # for img_batch, label_batch, _ in tqdm(test_dataset.loader, total=test_dataset.loader.__len__(), ncols=0):
        for img_batch, label_batch, _, _ in test_dataset.loader:
            b = img_batch.size(0)
            img_batch = img_batch.cuda()        # (b, 3, h, w)
            label_batch = label_batch.cuda()    # (b, 1, h, w)
            pre_batch = net(img_batch,)


            #pre_batch = F.interpolate(pre_batch, size=(label_batch.size(2), label_batch.size(3)))
            pre_batch = torch.sigmoid(pre_batch)


            pre_batch[pre_batch > 0.5] = 1
            pre_batch[pre_batch <= 0.5] = 0
            f1_score, iou_score = calc_metric(pre_batch.long(), label_batch.long())
            

            if iou_score_all is None:
                f1_score_all = f1_score
                iou_score_all = iou_score
            else:
                f1_score_all = torch.cat([f1_score_all, f1_score], 0)
                iou_score_all = torch.cat([iou_score_all, iou_score], 0)


        if f1_score_all.mean() > BEST_F1:
            BEST_F1 = f1_score_all.mean()
            BEST_IOU = iou_score_all.mean()
            BEST_EPOCH = epoch_index + 1
            net.cpu()
            torch.save(net.state_dict(), os.path.join(cfg.weight_dir, 'BestNet.pth'))#% (epoch_index + 1)
            net.cuda()


        line = '[%s] [Epoch %d LR %.6f] [Loss %.6f] [F1/IOU %.2f/%.2f] [BEST %d %.2f/%.2f] [cost %.2f seconds]' % (
                    cfg.exp_name, 
                    epoch_index + 1, 
                    opt.param_groups[0]['lr'], 
                    mean_loss, 
                    f1_score_all.mean() * 100, 
                    iou_score_all.mean() * 100, 
                    BEST_EPOCH, 
                    BEST_F1 * 100, 
                    BEST_IOU * 100,
                    timeit.default_timer()-start
                )
        print(line)
        start = timeit.default_timer()

        with open(cfg.log_path, 'a') as fp:
            fp.write(line + '\n')
        
    else:
        line = '[%s] [Epoch %d LR %.6f] [Loss %.6f]  ' % (
                    cfg.exp_name, 
                    epoch_index + 1, 
                    opt.param_groups[0]['lr'], 
                    mean_loss.mean(),

                )
        with open(cfg.log_path, 'a') as fp:
            fp.write(line + '\n')
        #start = timeit.default_timer()

    # save
    torch.save(net.state_dict(), os.path.join(cfg.weight_dir, 'NewestNet.pth'))
    # if (epoch_index + 1) % 100 == 0:
    #     net.cpu()
    #     torch.save(net.state_dict(), cfg.weight_path_template % (epoch_index + 1))
    #     net.cuda()
    writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], epoch_index)
    writer.add_scalar('Train_allloss', mean_loss.item(), epoch_index)
    writer.add_scalar('Test_Dice', f1_score_all.mean().item(), epoch_index)
