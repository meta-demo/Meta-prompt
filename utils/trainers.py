import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import time
from utils.helper import AverageMeter, mAP
from utils.asymmetric_loss import ProtypicalLoss
from torch.cuda.amp import autocast
import os
import json
from tqdm import tqdm
import numpy as np
import random
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from randaugment import RandAugment

def get_data(path, mode='json'):
    result = []
    with open(path, 'r') as src:
        if mode == 'json':
            for line in tqdm(src):
                line = json.loads(line)
                result.append(line)
        else:
            for line in tqdm(src):
                line = line.split('\n')[0]
                result.append(line)
    return result


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

from collections import defaultdict
def obtain_label(episode_labels, labels):
    i = 0
    label_dict = defaultdict(int)
    for label in episode_labels:
        label_dict[label] = i
        i += 1
    label_ids = []
    for label in labels:
        tmp = []
        for l in episode_labels:
            if l in label:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)
    return label_ids


def obtain_img_labels(root, split, support, query, episode_labels, img_size=224):
    
   

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    if split == 'train' or split == 'val':
        transform = train_transform
    elif split == "test":
        transform = test_transform
    else:
        raise ValueError('data split = %s is not supported in mscoco' % self.data_split)
    support_img, query_img = [], []
    support_label, query_label = [], []
    for line in support:
        file_name = line['file_name']
        img_path = os.path.join(root, split, 'images/' + file_name)
        img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)
        support_img.append(img)
        support_label.append(line['label'])
    for line in query:
        file_name = line['file_name']
        img_path = os.path.join(root, split, 'images/' + file_name)
        img = Image.open(img_path).convert('RGB')
        if transform is not None:
            img = transform(img)
        query_img.append(img)
        query_label.append(line['label'])
        
    support_ids = obtain_label(episode_labels, support_label)
    query_ids = obtain_label(episode_labels, query_label)
    
    support_img = torch.stack(support_img)
    query_img = torch.stack(query_img)
    support_ids = torch.tensor(support_ids)
    query_ids = torch.tensor(query_ids)
    
    return support_img, query_img, support_ids, query_ids


def train_coop(data_loader, model, optim, sched, args, cfg):
    batch_time = AverageMeter()
    miAP_batches = AverageMeter()
    maAP_batches = AverageMeter()
    losses = AverageMeter()
    acc_batches = AverageMeter()
    cacc_batches = AverageMeter()
    macro_p_batches, macro_r_batches, macro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()
    micro_p_batches, micro_r_batches, micro_f_batches = AverageMeter(), AverageMeter(), AverageMeter()
    
    
    # # switch to evaluate mode
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.image_encoder.train()
    else:
        model.module.prompt_learner.train()
        if cfg.TRAINER.FINETUNE_ATTN:
            model.module.image_encoder.attnpool.train()

        if cfg.TRAINER.FINETUNE_BACKBONE:
            model.module.image_encoder.train()

    criterion = ProtypicalLoss(cfg.TRAINER.DEVICEID, cfg.TRAINER.BETA, cfg.TRAINER.GAMMA)
    
    root = cfg.DATASET.ROOT 
    split = 'train'
    
    end = time.time()
    for i, (batch) in enumerate(data_loader):
        
        support, query, classnames = batch
        print(i, len(support), len(query))
        print(classnames)
        
        support_img, query_img, support_ids, query_ids = obtain_img_labels(root, split, support, query, classnames, args.input_size)
       
        support_size = support_img.shape[0]
        images = torch.cat((support_img, query_img), dim=0)
        target = torch.cat((support_ids, query_ids), dim=0)
       

        
        if torch.cuda.is_available():
            device = torch.device("cuda", cfg.TRAINER.DEVICEID)
        else:
            device = torch.device("cpu")
        images = images.to(device)
        target = target.to(device)
        
        # compute output
        with autocast():
            image_features, text_features, count_outputs, weights = model(classnames, images)
        
        loss, acc, c_acc, mi_ap, micro_p, micro_r, micro_f, \
            ma_ap, macro_p, macro_r, macro_f = criterion(image_features, text_features, count_outputs, weights, support_size, target)

       
        # update the network
        optim.zero_grad()
        loss.backward()
        optim.step()
        sched.step()

        losses.update(loss.item(), 1)
       

        print(f"Loss: {loss}, Acc: {acc}, Count Acc: {c_acc}")
        print(f"Macro mAP:{ma_ap}, Macro P: {macro_p}, Macro R: {macro_r}, Macro F1: {macro_f}")
        print(f"Micro mAP:{mi_ap}, Micro P: {micro_p}, Micro R: {micro_r}, Micro F1: {micro_f}")

        maAP_batches.update(ma_ap)
        miAP_batches.update(mi_ap)
        acc_batches.update(acc)
        cacc_batches.update(c_acc)
        micro_p_batches.update(micro_p)
        micro_r_batches.update(micro_r)
        micro_f_batches.update(micro_f)
        macro_p_batches.update(macro_p)
        macro_r_batches.update(macro_r)
        macro_f_batches.update(macro_f)
        
        batch_time.update(time.time()-end)
        end = time.time()
       
    

    return batch_time, losses, acc_batches, cacc_batches, \
            miAP_batches, micro_p_batches, micro_r_batches, micro_f_batches, \
            maAP_batches, macro_p_batches, macro_r_batches, macro_f_batches