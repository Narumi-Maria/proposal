import os
import math
import shutil
import numpy as np
from PIL import Image
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from config import opt
# from model.proposal_model import *
# from model.resnet_4ch import *
from proposal_model import *


class ImageDataset(Dataset):
    def __init__(self, istrain=True):
        self.istrain = istrain
        with open(opt.train_data_path if istrain else opt.test_data_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]
        self.labels = []
        self.images_path = []
        self.mask_path = []
        self.obj_box = []  # foreground box
        self.dic_name = []
        for row in reader:
            label = int(row[-3])
            image_path = row[-2]
            mask_path = row[-1]
            obj_box = eval(row[3])
            self.labels.append(label)
            self.images_path.append(opt.img_path + image_path)
            self.mask_path.append(opt.mask_path + mask_path)
            self.obj_box.append(obj_box)
            self.dic_name.append(image_path)

        self.img_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
        ])
        # reference box and depth feature
        if istrain:
            self.refer_box_dic = np.load(opt.box_dic_path)[
                ()]  # dict{'img_name':ndarray(5,6)} 6- left,top,bottom,right,class_id,conf_score
            self.depth_fea_dic = np.load(opt.depth_feats_path)[()]  # dict{'img_name':ndarray(1,256,256)}
        else:
            self.refer_box_dic = np.load(opt.test_box_dic_path)[()]
            self.depth_fea_dic = np.load(opt.test_depth_feats_path)[()]

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')
        mask = Image.open(self.mask_path[index]).convert('L')  # gray
        w = img.width
        h = img.height
        img = self.img_transform(img)
        mask = self.img_transform(mask)
        img_cat = torch.cat((img, mask), dim=0)

        label = self.labels[index]
        obj_box = torch.tensor(self.obj_box[index])
        refer_box = torch.from_numpy(self.refer_box_dic[self.dic_name[index]])
        depth_fea = torch.from_numpy(self.depth_fea_dic[self.dic_name[index].replace("/", "+")])

        return img_cat, label, obj_box, refer_box, depth_fea, w, h

    def __len__(self):
        return len(self.labels)


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        print("=== loading checkpoint '{}' ===".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=== done. also loaded optimizer from epoch {}) ===".format(last_epoch + 1))
            return best_prec, last_epoch, optimizer


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(optimizer)
    if opt.lr_scheduler_type == 'STEP':
        if epoch in opt.lr_scheduler_lr_epochs:
            lr *= opt.lr_scheduler_lr_mults
    elif opt.lr_scheduler_type == 'COSINE':
        ratio = epoch / opt.epochs
        lr = opt.lr_scheduler_min_lr + \
             (opt.base_lr - opt.lr_scheduler_min_lr) * \
             (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif opt.lr_scheduler_type == 'HTD':
        ratio = epoch / opt.epochs
        lr = opt.lr_scheduler_min_lr + \
             (opt.base_lr - opt.lr_scheduler_min_lr) * \
             (1.0 - math.tanh(
                 opt.lr_scheduler_lower_bound
                 + (opt.lr_scheduler_upper_bound
                    - opt.lr_scheduler_lower_bound)
                 * ratio)
              ) / 2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_data_loader():
    trainset = ImageDataset(istrain=True)
    testset = ImageDataset(istrain=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.num_workers)
    return train_loader, test_loader


if __name__ == "__main__":
    pass
    # train_loader, test_loader = get_data_loader()
    # train_loader = get_data_loader()
    # net = pro_net()
    # for batch_index, (img_cat, label, obj_box, refer_box, depth_fea, w, h) in enumerate(train_loader):
    #     net(img_cat, obj_box, refer_box, depth_fea, w, h)
    #     # print(img_cat, label, obj_box, refer_box, depth_fea, img_size)
