import torch
import os
from torch.utils import data
import numpy as np
from PIL import Image
import pdb
import torchvision.transforms.functional as F
import pywt


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
            #if self.cfg.gt_resize:
            label = label.resize(size=self.cfg.output_size, resample=Image.NEAREST)
            label = np.array(label)[:, :, np.newaxis]   # (h, w, 1)
            label[label > 0] = 1

        label[label > 0] = 1
        # ===================================================
        # ------------------- Normalization --------------------
        img = np.array(img) / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1)

        label = torch.FloatTensor(label).permute(2, 0, 1)
        return img, label, img_path

    def __len__(self):
        return self.img_path_list.__len__()


class TestDataSet(object):
    def __init__(self, config):
        self.cfg = config


        self.img_path_list = []
        self.label_path_list = []
        with open(config.test_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                # print(config.train_dir)
                img_dir = os.path.join(self.cfg.test_dir[:-9], line[0])
                label_dir = os.path.join(self.cfg.test_dir[:-9], line[1])
                print("test",img_dir, label_dir)
                assert os.path.exists(img_dir)

                self.img_path_list.append(img_dir)
                assert os.path.exists(label_dir)
                self.label_path_list.append(label_dir)

        self.dataset = DataSetDefine(self.img_path_list, self.label_path_list, self.cfg)
        self.loader = data.DataLoader(dataset=self.dataset, batch_size=self.cfg.test_batch_size, shuffle=False, num_workers=8)

        print('Test Batch:', self.loader.__len__())
        print('Test Sample:', self.dataset.__len__())


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    import config as cfg
    dataset = TestDataSet(cfg)
