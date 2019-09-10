import os
import csv
import numpy as np

dir_name = './logs_halfPipe/'
total_PSNR_val_array = []
total_SSIM_val_array = []
total_loss_train_gen = []
for file in os.listdir(dir_name):
    PSNR_val_array = []
    SSIM_val_array = []
    loss_train_gen = []
    if file.endswith('.out'):
        if "dataAug" in file:
            out_file = open(os.path.join(dir_name, file))
            loss_train_gen.append(file.split(".")[0]+"_loss_train")
            PSNR_val_array.append(file.split(".")[0]+"_PSNR_val")
            SSIM_val_array.append(file.split(".")[0]+"_SSIM_val")
            previous_line = None
            for current_line in out_file:
                if "1051/1051" in current_line:
                    split_line = current_line.split(" ")
                    loss_g = float(split_line[4])
                    loss_train_gen.append(loss_g)
                if "117/117" in current_line:
                    split_line = current_line.split(" ")
                    PSNR = float(split_line[7])
                    PSNR_val_array.append(PSNR)
                    SSIM = float(split_line[10].split(":")[0])
                    SSIM_val_array.append(SSIM)
            if loss_train_gen.__len__()>10 and PSNR_val_array.__len__()>10 and SSIM_val_array.__len__()>10:
                total_loss_train_gen.append(loss_train_gen)
                total_PSNR_val_array.append(PSNR_val_array)
                total_SSIM_val_array.append(SSIM_val_array)
        else:
            out_file = open(os.path.join(dir_name, file))
            loss_train_gen.append(file.split(".")[0]+"_loss_train")
            PSNR_val_array.append(file.split(".")[0]+"_PSNR_val")
            SSIM_val_array.append(file.split(".")[0]+"_SSIM_val")
            previous_line = None
            for current_line in out_file:
                if "2002/2002" in current_line:
                    split_line = current_line.split(" ")
                    loss_g = float(split_line[4])
                    loss_train_gen.append(loss_g)
                if "501/501" in current_line:
                    split_line = current_line.split(" ")
                    PSNR = float(split_line[7])
                    PSNR_val_array.append(PSNR)
                    SSIM = float(split_line[10].split(":")[0])
                    SSIM_val_array.append(SSIM)
            if loss_train_gen.__len__()>10 and PSNR_val_array.__len__()>10 and SSIM_val_array.__len__()>10:
                total_loss_train_gen.append(loss_train_gen)
                total_PSNR_val_array.append(PSNR_val_array)
                total_SSIM_val_array.append(SSIM_val_array)

with open('./metrics_SRGAN.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    total_acc = zip(*total_loss_train_gen, *total_PSNR_val_array, *total_SSIM_val_array)
    wr.writerows(total_acc)
