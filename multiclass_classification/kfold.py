import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from data_utils import ImageFolder_shuffle

# import seaborn as sn
# import pandas as pd
# import torchnet.meter.confusionmeter as cm

# Data augmentation and normalization for training
# Just normalization for validation & test

parser = argparse.ArgumentParser(description='multiclass classification')
parser.add_argument('--whole_pipe', default=False, action='store_true')
parser.add_argument('--upscale_factor', default=1, type=int, help='super resolution upscale factor')
parser.add_argument('--lambda_class', default=-1.0, type=float)
parser.add_argument('--classifier_name', default="resnet50", type=str)
parser.add_argument('--batch_size', default=1, type=int, help='train batch_size number')

opt = parser.parse_args()
UPSCALE_FACTOR = opt.upscale_factor
WHOLE_PIPE = opt.whole_pipe
LAMBDA_CLASS = opt.lambda_class
CLASSIFIER_NAME = opt.classifier_name
BATCH_SIZE = opt.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

switcher = {
    'resnet50': models.resnet50,
    'resnet18': models.resnet18
}
# Get the function from switcher dictionary
func = switcher[CLASSIFIER_NAME]

model_ft = func(pretrained=True)
classifier_name = CLASSIFIER_NAME

if UPSCALE_FACTOR == 1:
    data_dir = '../data/split_dataset'
else:
    if WHOLE_PIPE and LAMBDA_CLASS == -1.0:
        data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR)+"_"+classifier_name

    if not WHOLE_PIPE:
        data_dir = '../data/split_dataset_SRGAN_' + str(UPSCALE_FACTOR)

    if WHOLE_PIPE and LAMBDA_CLASS != -1.0:
        if LAMBDA_CLASS == 1.0 or LAMBDA_CLASS == 2.0:
            LAMBDA_CLASS = int(LAMBDA_CLASS)
        data_dir = '../data/split_dataset_wholePipe_' + str(UPSCALE_FACTOR)+'_'+classifier_name+\
                   "_lambda"+ str(LAMBDA_CLASS)

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
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])exit
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(161, 161)),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

###KFOLD
image_dataset = None
if data_dir == '../data/split_dataset':
    image_dataset = ImageFolder_shuffle(data_dir+"/train", transform=data_transforms["train"])
elif UPSCALE_FACTOR > 1 and ('SRGAN' in data_dir or 'wholePipe' in data_dir):
    image_dataset = ImageFolder_shuffle(data_dir+"/train", transform=transforms.Compose([transforms.ToTensor()]))
    # image_dataset = ImageFolder_shuffle(data_dir, data_transforms["train"])

print(data_dir)
kf = KFold(n_splits=5)
print(kf.get_n_splits(image_dataset))
print(classifier_name)
print("whole pipe {}".format(WHOLE_PIPE))
print("LAMBDA {}".format(LAMBDA_CLASS))
class_names = image_dataset.classes
print(class_names)
####


# lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []


# Train the model
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                # inputs.cuda()
                # labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs.data[0])
                    # print(labels.data[0])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # For graph generation
            if phase == "train":
                train_loss.append(running_loss / dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "val":
                val_loss.append(running_loss / dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # for printing
            if phase == "train":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "val":
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, best_model_wts


# Using a model pre-trained on ImageNet and replacing it's final linear layer

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))


# for VGG16_BN
# model_ft = models.vgg16_bn(pretrained=True)
# model_ft.classifier[6].out_features = 8

model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#####KFOLD
for idx, (train_index, val_index) in enumerate(kf.split(image_dataset)):
    print("KFOLD %d" % idx)
    dataloaders_type = dict()
    dataset_sizes_dict = dict()
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    dataloaders_type["train"] = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE,
                                                            shuffle=False, num_workers=4, sampler=train_sampler)
    dataloaders_type["val"] = torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                                          shuffle=False, num_workers=4, sampler=val_sampler)
    dataset_sizes_dict["train"] = len(train_index)
    dataset_sizes_dict["val"] = len(val_index)
    model_ft, best_model_wts = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders_type,
                                           dataset_sizes_dict, num_epochs=25)
    ##Save weights
    if idx == kf.get_n_splits(image_dataset)-1:
        path = "weights/"
        if not os.path.exists(path):
            os.makedirs(path)
        if UPSCALE_FACTOR == 1:
            torch.save(best_model_wts, path + classifier_name+"_best_model_kfold.pth")
        else:
            if WHOLE_PIPE and LAMBDA_CLASS == -1.0:
                torch.save(best_model_wts, path + classifier_name+"_best_model_kfold_wholePipe_" + str(UPSCALE_FACTOR) + ".pth")

            if not WHOLE_PIPE:
                torch.save(best_model_wts, path + classifier_name+"_best_model_kfold_SRGAN_" + str(UPSCALE_FACTOR) + ".pth")

            if WHOLE_PIPE and LAMBDA_CLASS != -1.0:
                if LAMBDA_CLASS == 1.0 or LAMBDA_CLASS == 2.0:
                    LAMBDA_CLASS = int(LAMBDA_CLASS)
                torch.save(best_model_wts, path + classifier_name + "_best_model_kfold_wholePipe_" +
                           str(UPSCALE_FACTOR) + '_' + classifier_name + "_lambda" + str(LAMBDA_CLASS) + ".pth")
