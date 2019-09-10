import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision.utils as utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models, transforms
import numpy as np
import sys
import copy
import pytorch_ssim
from data_utils import *
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--hr_size', default=322, type=int, help='training images hr size')
parser.add_argument('--upscale_factor', default=2, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--batch_size', default=1, type=int, help='train batch_size number')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--data_aug', default=False, action='store_true')
parser.add_argument('--whole_pipe', default=False, action='store_true')
parser.add_argument('--lambda_class', default=False, action='store_true')

opt = parser.parse_args()

HR_SIZE = opt.hr_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
BATCH_SIZE = opt.batch_size
DATA_AUG = opt.data_aug
WHOLE_PIPE = opt.whole_pipe
LAMBDA_CLASS = opt.lambda_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_name(0)


def train_half_pipe():
    if DATA_AUG:
        train_set = TrainDatasetFromFolder('/home/mrey/ESA/Dataset/Step2-SuperresolutionWhale/split_SRdataset/train'
                                           ,hr_size=HR_SIZE, upscale_factor=UPSCALE_FACTOR)
    else:
        train_set = TrainDatasetFromFolder('/home/mrey/ESA/Dataset/Step2-SuperresolutionWhale/converted_jpg',
                                           hr_size=HR_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('/home/mrey/ESA/Dataset/Step2-SuperresolutionWhale/split_SRdataset/test',
                                   hr_size=HR_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    best_netG = copy.deepcopy(netG.state_dict())
    best_netD = copy.deepcopy(netD.state_dict())
    old_PSNR = 0.0
    old_SSIM = 0.0

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:

            # print('')
            # print([data_i.shape for data_i in data.data])
            # print([data_i.shape for data_i in target.data])
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img, 0, 0.0)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img, 0, 0.0)
            running_results['g_loss'] += g_loss.data * batch_size
            d_loss = 1 - real_out + fake_out
            running_results['d_loss'] += d_loss.data * batch_size
            running_results['d_score'] += real_out.data * batch_size
            running_results['g_score'] += fake_out.data * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = Variable(val_lr, volatile=True)
            hr = Variable(val_hr, volatile=True)
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).data
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0))])
        # val_images = torch.stack(val_images)
        # val_images = torch.chunk(val_images, val_images.size(0) // 15)
        # val_save_bar = tqdm(val_images, desc='[saving training results]')
        # index = 1
        # for image in val_save_bar:
        #     image = utils.make_grid(image, nrow=3, padding=5)
        #     if DATA_AUG:
        #         utils.save_image(image, out_path + 'dataAug_epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     else:
        #         utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     index += 1

        # SAVE model parameters
        if valing_results['psnr'] > old_PSNR and valing_results['ssim'] > old_SSIM:
            old_PSNR = valing_results['psnr']
            old_SSIM = valing_results['ssim']
            best_netG = copy.deepcopy(netG.state_dict())
            best_netD = copy.deepcopy(netD.state_dict())
            out_folder = 'epochs/weights_'+str(UPSCALE_FACTOR)+'_dataAug_halfPipe/'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            torch.save(best_netG, out_folder + 'best_netG.pth')
            torch.save(best_netD, out_folder + 'best_netD.pth')

        if epoch % 10 == 0 and epoch != 0:
            if DATA_AUG:
                out_folder = 'epochs/weights_'+str(UPSCALE_FACTOR)+'_dataAug_halfPipe/'
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                torch.save(netG.state_dict(), out_folder+'netG_dataAug_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
                torch.save(netD.state_dict(), out_folder+'netD_dataAug_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
            else:
                out_folder = 'epochs/weights_' + str(UPSCALE_FACTOR)+'_halfPipe/'
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                torch.save(netG.state_dict(), out_folder+'netG_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
                torch.save(netD.state_dict(), out_folder+'netD_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics_halfPipe/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            if DATA_AUG:
                data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_dataAug_train_results.csv', index_label='Epoch')
            else:
                data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')


def train_whole_pipe():
    data_dir_lr = '/home/mrey/ESA/Dataset/split_dataset/'
    data_dir_hr = '/home/mrey/ESA/Dataset/split_dataset_SRGAN_'+str(UPSCALE_FACTOR)+os.sep
    train_set = ImageFolderWithPaths_train(data_dir_hr+"train"+os.sep, data_dir_lr+"train", HR_SIZE, UPSCALE_FACTOR)
    val_set = ImageFolderWithPaths_val(data_dir_hr+"test"+os.sep, data_dir_lr+"test", HR_SIZE, UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    class_names = train_set.classes
    print(train_set.classes)
    print(val_set.classes)

    netD_weights = "epochs/weights_halfPipe/weights_"+str(UPSCALE_FACTOR)\
                   + "_dataAug/netD_dataAug_epoch_"+str(UPSCALE_FACTOR)+"_050.pth"
    netG_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) \
                   + "_dataAug/netG_dataAug_epoch_" + str(UPSCALE_FACTOR) + "_050.pth"
    # netD_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) + "_dataAug/best_netD.pth"
    # netG_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) + "_dataAug/best_netG.pth"

    netG = Generator(UPSCALE_FACTOR)
    netG.load_state_dict(torch.load(netG_weights))
    netD = Discriminator()
    netD.load_state_dict(torch.load(netD_weights))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    ####CLASSIFIER
    classifier = models.resnet50(pretrained=True)
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, len(class_names))
    classifier.name = 'resnet50'
    classifier.cuda()
    # if UPSCALE_FACTOR == 1:
    #     weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/best_model.pth"
    # else:
    #     weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/best_model_SRGAN_"\
    #                          + str(UPSCALE_FACTOR)+".pth"
    weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/"+\
                         classifier.name+"_best_model_kfold.pth"
    classifier.load_state_dict(torch.load(weights_class_path))
    criterion_classifier = nn.CrossEntropyLoss()
    classifier.eval()
    print(classifier.name)
    print("upscale factor %d" % UPSCALE_FACTOR)
    print("HR size %d" % HR_SIZE)
    #############

    best_netG = copy.deepcopy(netG.state_dict())
    best_netD = copy.deepcopy(netD.state_dict())
    old_PSNR = 0
    old_SSIM = 0

    miss_classifications = []

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target, label in train_bar:

            # print('')
            # print([data_i.shape for data_i in data.data])
            # print([data_i.shape for data_i in target.data])
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
                label = label.to(device)
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            classifier_outputs = classifier(z)
            _, preds = torch.max(classifier_outputs, 1)
            if preds != label:
                miss_classifications.append([preds, label])
                # print(str(preds)+"--"+str(label))
            loss_classifier = criterion_classifier(classifier_outputs, label)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img, loss_classifier, 0.001)
            g_loss.backward()
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img, loss_classifier, 0.001)
            running_results['g_loss'] += g_loss.data * batch_size
            d_loss = 1 - real_out + fake_out
            running_results['d_loss'] += d_loss.data * batch_size
            running_results['d_score'] += real_out.data * batch_size
            running_results['g_score'] += fake_out.data * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = Variable(val_lr, volatile=True)
            hr = Variable(val_hr, volatile=True)
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).data
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0))])
        # val_images = torch.stack(val_images)
        # val_images = torch.chunk(val_images, val_images.size(0) // 15)
        # val_save_bar = tqdm(val_images, desc='[saving training results]')
        # index = 1
        # for image in val_save_bar:
        #     image = utils.make_grid(image, nrow=3, padding=5)
        #     if DATA_AUG:
        #         utils.save_image(image, out_path + 'dataAug_epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     else:
        #         utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #     index += 1

        # SAVE model parameters
        if valing_results['psnr'] > old_PSNR and valing_results['ssim'] > old_SSIM:
            old_PSNR = valing_results['psnr']
            old_SSIM = valing_results['ssim']
            best_netG = copy.deepcopy(netG.state_dict())
            best_netD = copy.deepcopy(netD.state_dict())
            out_folder = 'epochs/weights_' + str(UPSCALE_FACTOR) + '_' + str(classifier.name) + '_wholePipe/'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            torch.save(best_netG, out_folder + 'best_netG.pth')
            torch.save(best_netD, out_folder + 'best_netD.pth')

        if epoch % 5 == 0 and epoch != 0:
            out_folder = 'epochs/weights_' + str(UPSCALE_FACTOR)+'_'+str(classifier.name)+'_wholePipe/'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            torch.save(netG.state_dict(), out_folder+'netG_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), out_folder+'netD_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 5 == 0 and epoch != 0:
            out_path = 'statistics_wholePipe/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_'+str(classifier.name)+'_train_results.csv', index_label='Epoch')

    # print(miss_classifications)


def train_lambda_class():
    data_dir_lr = '/home/mrey/ESA/Dataset/split_dataset/'
    data_dir_hr = '/home/mrey/ESA/Dataset/split_dataset_SRGAN_'+str(UPSCALE_FACTOR)+os.sep
    train_set = ImageFolderWithPaths_train(data_dir_hr+"train"+os.sep, data_dir_lr+"train", HR_SIZE, UPSCALE_FACTOR)
    val_set = ImageFolderWithPaths_val(data_dir_hr+"test"+os.sep, data_dir_lr+"test", HR_SIZE, UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    class_names = train_set.classes
    print(train_set.classes)
    print(val_set.classes)

    netD_weights = "epochs/weights_halfPipe/weights_"+str(UPSCALE_FACTOR)\
                   + "_dataAug/netD_dataAug_epoch_"+str(UPSCALE_FACTOR)+"_050.pth"
    netG_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) \
                   + "_dataAug/netG_dataAug_epoch_" + str(UPSCALE_FACTOR) + "_050.pth"
    # netD_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) + "_dataAug/best_netD.pth"
    # netG_weights = "epochs/weights_halfPipe/weights_" + str(UPSCALE_FACTOR) + "_dataAug/best_netG.pth"

    netG = Generator(UPSCALE_FACTOR)
    netG.load_state_dict(torch.load(netG_weights))
    netD = Discriminator()
    netD.load_state_dict(torch.load(netD_weights))

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    ####CLASSIFIER
    classifier = models.resnet50(pretrained=True)
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, len(class_names))
    classifier.name = 'resnet50'
    classifier.cuda()
    # if UPSCALE_FACTOR == 1:
    #     weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/best_model.pth"
    # else:
    #     weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/best_model_SRGAN_"\
    #                          + str(UPSCALE_FACTOR)+".pth"
    weights_class_path = "/home/mrey/ESA/pruebas/multiclass_classification/weights/"+\
                         classifier.name+"_best_model_kfold.pth"
    classifier.load_state_dict(torch.load(weights_class_path))
    criterion_classifier = nn.CrossEntropyLoss()
    classifier.eval()
    print(classifier.name)
    print("upscale factor %d" % UPSCALE_FACTOR)
    print("HR size %d" % HR_SIZE)
    #############

    best_netG = copy.deepcopy(netG.state_dict())
    best_netD = copy.deepcopy(netD.state_dict())
    old_PSNR = 0
    old_SSIM = 0
    basic_netG = copy.deepcopy(netG.state_dict())
    basic_netD = copy.deepcopy(netD.state_dict())

    miss_classifications = []

    lambda_values = [2, 1, 0.1, 0.01]

    for lambda_class in lambda_values:
        results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        netG.load_state_dict(basic_netG)
        netD.load_state_dict(basic_netD)
        print("CHOSE LAMBDA {}".format(lambda_class))
        for epoch in range(1, NUM_EPOCHS + 1):
            train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

            netG.train()
            netD.train()
            for data, target, label in train_bar:

                # print('')
                # print([data_i.shape for data_i in data.data])
                # print([data_i.shape for data_i in target.data])
                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                if torch.cuda.is_available():
                    real_img = real_img.cuda()
                    label = label.to(device)
                z = Variable(data)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_img = netG(z)

                classifier_outputs = classifier(z)
                _, preds = torch.max(classifier_outputs, 1)
                if preds != label:
                    miss_classifications.append([preds, label])
                    # print(str(preds)+"--"+str(label))
                loss_classifier = criterion_classifier(classifier_outputs, label)

                netD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()
                d_loss = 1 - real_out + fake_out
                d_loss.backward(retain_graph=True)
                optimizerD.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                netG.zero_grad()
                g_loss = generator_criterion(fake_out, fake_img, real_img, loss_classifier, float(lambda_class))
                g_loss.backward()
                optimizerG.step()
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()

                g_loss = generator_criterion(fake_out, fake_img, real_img, loss_classifier, float(lambda_class))
                running_results['g_loss'] += g_loss.data * batch_size
                d_loss = 1 - real_out + fake_out
                running_results['d_loss'] += d_loss.data * batch_size
                running_results['d_score'] += real_out.data * batch_size
                running_results['g_score'] += fake_out.data * batch_size

                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))

            netG.eval()
            out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = Variable(val_lr, volatile=True)
                hr = Variable(val_hr, volatile=True)
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).data
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            # index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     if DATA_AUG:
            #         utils.save_image(image, out_path + 'dataAug_epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     else:
            #         utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1

            # SAVE model parameters
            if epoch % 5 == 0 and epoch != 0:
                out_folder = 'epochs/weights_' + str(UPSCALE_FACTOR)+'_'+str(classifier.name)+'_lambda' + \
                             str(lambda_class)+'_wholePipe/'
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                torch.save(netG.state_dict(), out_folder+'netG_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
                torch.save(netD.state_dict(), out_folder+'netD_epoch_%d_%03d.pth' % (UPSCALE_FACTOR, epoch))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

            if epoch % 5 == 0 and epoch != 0:
                out_path = 'statistics_wholePipe/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                          'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))
                data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_'+str(classifier.name)+'_lambda' +
                                  str(lambda_class)+'_train_results.csv', index_label='Epoch')

    # print(miss_classifications)


if __name__ == "__main__":
    if WHOLE_PIPE and not LAMBDA_CLASS:
        train_whole_pipe()
    if not WHOLE_PIPE:
        train_half_pipe()
    if WHOLE_PIPE and LAMBDA_CLASS:
        train_lambda_class()