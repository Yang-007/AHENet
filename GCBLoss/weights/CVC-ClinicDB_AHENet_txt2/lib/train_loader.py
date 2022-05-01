import torch
import os
from torch.utils import data
import numpy as np
from PIL import Image
import pdb
from scipy import ndimage
import random
import torchvision.transforms.functional as F
import pywt
from torchvision import transforms


class DataSetDefine(data.Dataset):
    def __init__(self, img_path_list, label_path_list, config):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.cfg = config
        assert self.img_path_list.__len__() == self.label_path_list.__len__()

    def __getitem__(self, index):
        """
        :param index:
        :return: (1, 3, h, w), (1, 3, h, w)
        """
        img_path = self.img_path_list[index]
        label_path = self.label_path_list[index]


        with Image.open(img_path) as img:
            img = img.convert('RGB')    # (h, w, 3)
            img = img.resize(size=self.cfg.output_size)


        with Image.open(label_path) as label:
            label = label.convert('L')  # (h, w)
            label = label.resize(size=self.cfg.output_size, resample=Image.NEAREST)


        #------------------- Random Flip+Random rot90 ---------------
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            img = F.rotate(img, k*90.0, F.Image.BILINEAR, False, None, 0)
            label = F.rotate(label, k*90.0, F.Image.NEAREST, False, None, 0)

            if random.random() > 0.5:
                img = F.hflip(img)
                label = F.hflip(label)
            else:
                img = F.vflip(img)
                label = F.vflip(label)
        # ------------------ Random Rotate ------------------
        angle = transforms.RandomRotation.get_params([-20, +20])
        img = F.rotate(img, angle, F.Image.BILINEAR, False, None, 0)
        label = F.rotate(label, angle, F.Image.NEAREST, False, None, 0)

        # ------------------- Normalization --------------------
        img = np.array(img) / 255.0
        label = np.array(label)[:, :, np.newaxis]   # (h, w, 1)
        label[label > 0] = 1
        img = torch.FloatTensor(img).permute(2, 0, 1)    
        label = torch.FloatTensor(label).permute(2, 0, 1)

        # ------------------ Random Erasing --------------------
        if random.random() > 0.5:
            x, y, h, w, v = transforms.RandomErasing.get_params(img, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=[0])
            img = F.erase(img, x, y, h, w, v, False)
            label = F.erase(label, x, y, h, w, v, False)

        return img, label

    def __len__(self):
        return self.img_path_list.__len__()


class TrainDataSet(object):
    def __init__(self, config):
        self.cfg = config

        self.img_path_list = []
        self.label_path_list = []
        with open(config.train_dir,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split()
                #print(config.train_dir)
                img_dir = os.path.join(config.train_dir[:-10], line[0])
                label_dir = os.path.join(config.train_dir[:-10], line[1])
                #print(img_dir,label_dir)
                assert os.path.exists(img_dir)

                self.img_path_list.append(img_dir)
                print('label_dir:',label_dir)
                assert os.path.exists(label_dir)
                self.label_path_list.append(label_dir)

        self.dataset = DataSetDefine(self.img_path_list, self.label_path_list, self.cfg)
        self.loader = data.DataLoader(dataset=self.dataset, batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=8)

        print('Train Batch:', self.loader.__len__())
        print('Train Sample:', self.dataset.__len__())


if __name__ == '__main__':
    import config1 as cfg
    dataset = TrainDataSet(cfg)

    for img_batch, label_batch in dataset.loader:
        print(img_batch.size(), label_batch.size())
        print(img_batch.min().item(), img_batch.max().item(), label_batch.unique())
        # break
