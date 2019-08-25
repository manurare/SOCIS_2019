import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, models, transforms
from data_utils import *
import torch.nn as nn
import torchvision.utils as utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models
import numpy as np
import sys
from model import Generator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--generate_dataset', default=False, action='store_true')
parser.add_argument('--whole_pipe', default=False, action='store_true')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
GEN_DATASET = opt.generate_dataset
WHOLE_PIPE = opt.whole_pipe
if WHOLE_PIPE:
    MODEL_NAME = 'netG_epoch_'+str(UPSCALE_FACTOR)+'_050.pth'
else:
    MODEL_NAME = 'best_netG.pth'

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(device.index))
classifier_name = "resnet18"

if GEN_DATASET:
    data_dir = '/home/manuelrey/ESA/Dataset/split_dataset'
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    netG = Generator(UPSCALE_FACTOR)
    if WHOLE_PIPE:
        netG.load_state_dict(torch.load('epochs/weights_wholePipe/weights_'+str(UPSCALE_FACTOR)+'_'+classifier_name
                                        + '_wholePipe/' + MODEL_NAME))
    else:
        netG.load_state_dict(torch.load('epochs/weights_halfPipe/weights_'+str(UPSCALE_FACTOR)+'_dataAug/' + MODEL_NAME))
    if torch.cuda.is_available():
        netG.to(device)
    netG.eval()

    for name_dataloader in dataloaders:
        if WHOLE_PIPE:
            path = '/home/manuelrey/ESA/Dataset/split_dataset_wholePipe_'+str(UPSCALE_FACTOR)+'_'+classifier_name+os.sep
        else:
            path = '/home/manuelrey/ESA/Dataset/split_dataset_SRGAN_'+str(UPSCALE_FACTOR)+os.sep
        if not os.path.exists(path):
            os.makedirs(path)
        for lr_image, label, tuple_name, tuple_folder in dataloaders[name_dataloader]:
            folder = tuple_folder[0]
            name = tuple_name[0]
            if torch.cuda.is_available():
                lr_image = lr_image.to(device)
            sr_image = netG(lr_image)
            complete_path = path+folder+os.sep
            if not os.path.exists(complete_path):
                os.makedirs(complete_path)
            out_img = ToPILImage()(sr_image[0].data.cpu())
            out_img.save(complete_path+name)
            print("saved--------------->"+name)


