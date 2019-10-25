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
from PIL import Image

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--generate_dataset', default=False, action='store_true')
parser.add_argument('--whole_pipe', default=False, action='store_true')
parser.add_argument('--lambda_class', default=-1.0, type=float)

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
GEN_DATASET = opt.generate_dataset
WHOLE_PIPE = opt.whole_pipe
LAMBDA_CLASS = opt.lambda_class

if WHOLE_PIPE and LAMBDA_CLASS == -1.0:
    if UPSCALE_FACTOR == 2:
        MODEL_NAME = 'netG_epoch_'+str(UPSCALE_FACTOR)+'_030.pth'
    if UPSCALE_FACTOR == 4:
        MODEL_NAME = 'netG_epoch_'+str(UPSCALE_FACTOR)+'_020.pth'

if not WHOLE_PIPE:
    MODEL_NAME = 'best_netG.pth'

if WHOLE_PIPE and LAMBDA_CLASS != -1.0:
    MODEL_NAME = 'netG_epoch_' + str(UPSCALE_FACTOR) + '_020.pth'

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
classifier_name = "resnet50"
print(classifier_name)
print(UPSCALE_FACTOR)

if GEN_DATASET:
    data_dir = '../data/split_dataset/'
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    netG = Generator(UPSCALE_FACTOR)

    if WHOLE_PIPE and LAMBDA_CLASS == -1.0:
        netG.load_state_dict(torch.load('epochs/weights_wholePipe/weights_'+str(UPSCALE_FACTOR)+'_'+classifier_name
                                        + '_wholePipe/' + MODEL_NAME))
    if not WHOLE_PIPE:
        netG.load_state_dict(torch.load('epochs/weights_halfPipe/weights_'+str(UPSCALE_FACTOR)+'_dataAug/' + MODEL_NAME))

    if WHOLE_PIPE and LAMBDA_CLASS != -1.0:
        if LAMBDA_CLASS == 1.0 or LAMBDA_CLASS == 2.0:
            LAMBDA_CLASS = int(LAMBDA_CLASS)
        netG.load_state_dict(torch.load('epochs/weights_' + str(UPSCALE_FACTOR) + '_' +
                                        classifier_name + '_lambda' + str(LAMBDA_CLASS) + '_wholePipe/' + MODEL_NAME))

    if torch.cuda.is_available():
        netG.to(device)
    netG.eval()

    for name_dataloader in dataloaders:
        if WHOLE_PIPE and LAMBDA_CLASS == -1.0:
            path = '../data/split_dataset_wholePipe_'+str(UPSCALE_FACTOR)+'_'+classifier_name+os.sep
        if not WHOLE_PIPE:
            path = '../data/split_dataset_SRGAN_'+str(UPSCALE_FACTOR)+os.sep
        if WHOLE_PIPE and LAMBDA_CLASS != -1.0:
            path = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR) + '_' + classifier_name + \
                   '_lambda' + str(LAMBDA_CLASS) + os.sep

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

else:
    data_dir = '../data/split_dataset/'
    out_folder = '../data/split_dataset_BICUBIC_'+str(UPSCALE_FACTOR)+os.sep
    for dp, dn, filenames in os.walk(data_dir):
        if len(filenames) != 0:
            for file in filenames:
                path_to_save = out_folder+dp.split("/")[-2]+os.sep+dp.split("/")[-1]+os.sep
                img = Image.open(dp+os.sep+file)
                img = img.resize((161*UPSCALE_FACTOR,161*UPSCALE_FACTOR), Image.BICUBIC)
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                img.save(path_to_save+file)
