import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import KFold
from sklearn import metrics

import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm

# Data augmentation and normalization for training
# Just normalization for validation & test

parser = argparse.ArgumentParser(description='multiclass classification')
parser.add_argument('--whole_pipe', default=False, action='store_true')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--lambda_class', default=-1.0, type=float)
parser.add_argument('--dyn_lambda', default=False, action='store_true')
parser.add_argument('--classifier_name', default="resnet50", type=str)
parser.add_argument('--isolate_class', default=-1, type=int, help='DS=0, Roc=1, Sh=2, SSh=3, Wat=4, Wha=5, clo=6')

opt = parser.parse_args()
UPSCALE_FACTOR = opt.upscale_factor
WHOLE_PIPE = opt.whole_pipe
LAMBDA_CLASS = opt.lambda_class
CLASSIFIER_NAME = opt.classifier_name
ISOLATE_CLASS = opt.isolate_class
DYN_LAMBDA = opt.dyn_lambda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

####CLASSIFIER
switcher = {
    'resnet50': models.resnet50,
    'resnet18': models.resnet18
}
# Get the function from switcher dictionary
func = switcher[CLASSIFIER_NAME]

model_ft = func(pretrained=True)
model_ft.name = CLASSIFIER_NAME
print(model_ft.name)
class_names = ['Dynamic_ship', 'Rocks', 'Ship', 'Static_ship', 'Water', 'Whale', 'clouds']

data_dir = None
if UPSCALE_FACTOR == 1:
    data_dir = '../data/split_dataset'
else:
    if not WHOLE_PIPE:
        data_dir = '../data/split_dataset_SRGAN_'+str(UPSCALE_FACTOR)

    if WHOLE_PIPE:
        if DYN_LAMBDA:
            if ISOLATE_CLASS != -1:
                data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR) + '_' + \
                           model_ft.name + '_dynLambda_class_' + str(class_names[ISOLATE_CLASS]) + os.sep
            else:
                data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR) + '_' + \
                           model_ft.name + '_dynLambda_allClasses' + os.sep
        else:
            if ISOLATE_CLASS != -1:
                if LAMBDA_CLASS >= 1:
                    LAMBDA_CLASS = int(LAMBDA_CLASS)
                data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR)+'_'+model_ft.name +\
                           "_lambda" + str(LAMBDA_CLASS)+"_class_" + str(class_names[ISOLATE_CLASS])
            else:
                if LAMBDA_CLASS >= 1:
                    LAMBDA_CLASS = int(LAMBDA_CLASS)
                data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR)+'_'+model_ft.name +\
                           "_lambda" + str(LAMBDA_CLASS) + "_allClasses"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(161*UPSCALE_FACTOR, 161*UPSCALE_FACTOR)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])exit
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

test_dataset = None
if data_dir == '../data/split_dataset':
    test_dataset = datasets.ImageFolder(data_dir+"/test", transform=data_transforms["test"])
    train_dataset = datasets.ImageFolder(data_dir+"/train", transform=data_transforms["train"])
elif UPSCALE_FACTOR > 1 and ('SRGAN' in data_dir or 'wholePipe' in data_dir):
    test_dataset = datasets.ImageFolder(data_dir+"/test", transform=transforms.Compose([transforms.ToTensor()]))
    train_dataset = datasets.ImageFolder(data_dir+"/train", transform=transforms.Compose([transforms.ToTensor()]))
    # image_dataset = ImageFolder_shuffle(data_dir, data_transforms["train"])

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False , num_workers=4)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False , num_workers=4)
class_names = test_dataset.classes
print(data_dir)
print(class_names)

#lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

#For resnet18
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))


#for VGG16_BN
#model_ft = models.vgg16_bn(pretrained=True)
#model_ft.classifier[6].out_features = 8

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if UPSCALE_FACTOR == 1:
    model_ft.load_state_dict(torch.load("weights/" + model_ft.name + "_best_model_kfold.pth"))
else:
    model_ft.load_state_dict(torch.load("weights/"+model_ft.name+"_best_model_kfold_SRGAN_" +
                                        str(UPSCALE_FACTOR)+".pth"))

model_ft.eval()

#Get the confusion matrix for testing data
y_true = []
y_pred = []
confusion_matrix = cm.ConfusionMeter(len(class_names))
class_correct = list(0. for i in range(len(class_names)))
class_total = list(0. for i in range(len(class_names)))
scores = list(0. for i in range(len(class_names)))
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs, 1)
        point = (predicted == labels)
        for j in range(len(labels)):
            class_correct[labels[j].item()] += point[j].item()
            class_total[labels[j].item()] += 1
            scores[labels[j].item()] += outputs[0][labels[j].item()].cpu().numpy()
            y_true.append(labels[j].item())
            y_pred.append(predicted[j].item())
        confusion_matrix.add(predicted, labels)
    print(confusion_matrix.conf)
print("")
for i in range(len(class_names)):
    print('Accuracy of %5s : %2d %% --- Score: %3f' % (
        class_names[i], 100 * class_correct[i] / class_total[i], scores[i]))

# print("")
# print(metrics.confusion_matrix(y_true, y_pred))
print("")
print(metrics.classification_report(y_true, y_pred, digits=4, target_names=class_names))

