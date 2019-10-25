import os
import csv
import numpy as np

dir_name = '../multiclass_classification/logs'
total_acc_train_array = []
total_loss_train_array = []
total_acc_val_array = []
total_loss_val_array = []
for file in os.listdir(dir_name):
    loss_train_array = []
    loss_val_array = []
    acc_train_array = []
    acc_val_array = []
    if file.endswith('.out'):
        out_file = open(os.path.join(dir_name, file))
        acc_train_array.append(file+"_acc_train")
        loss_train_array.append(file+"_loss_train")
        acc_val_array.append(file+"_acc_val")
        loss_val_array.append(file+"_loss_val")
        for line in out_file:
            if "train" in line:
                split_line = line.split(" ")
                acc_train = float(split_line[-1])
                acc_train_array.append(acc_train)
                loss_train = float(split_line[-3])
                loss_train_array.append(loss_train)
            if "val" in line and "Best" not in line:
                split_line = line.split(" ")
                acc_val = float(split_line[-1])
                acc_val_array.append(acc_val)
                loss_val = float(split_line[-3])
                loss_val_array.append(loss_val)
        total_acc_train_array.append(acc_train_array)
        total_acc_val_array.append(acc_val_array)
        total_loss_train_array.append(loss_train_array)
        total_loss_val_array.append(loss_val_array)

with open('../multiclass_classification/metrics.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    total_acc = zip(*total_acc_train_array, *total_acc_val_array, )
    wr.writerows(total_acc)
