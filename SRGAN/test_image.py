import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
import sys
from data_utils import *
import os
import torch.utils.data
import torchvision.utils as utils

from model import Generator


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--test_set_dir', type=str, help='test set directory')
parser.add_argument('--whole_pipe', default=False, action='store_true')
parser.add_argument('--model_name', default='netG_epoch_2_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

#/home/manuelrey/ESA/Dataset/Step1-PresenceWhale/Whale/a_61.jpg
# /home/manuelrey/ESA/Dataset/Step1-PresenceWhale/whale_test


UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
TEST_DIR = opt.test_set_dir
WHOLE_PIPE = opt.whole_pipe

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    if WHOLE_PIPE:
        model.load_state_dict(torch.load('epochs/weights_2_wholePipe/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/weights_halfPipe/weights_'+str(UPSCALE_FACTOR)+"_dataAug/"
                                         + MODEL_NAME))
else:
    if WHOLE_PIPE:
        model.load_state_dict(torch.load('epochs/weights_2_wholePipe/' + MODEL_NAME
                                         , map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load('epochs/weights_halfPipe/weights_'+str(UPSCALE_FACTOR)+"_dataAug/"
                                         + MODEL_NAME, map_location=lambda storage, loc: storage))


if TEST_DIR is not None and IMAGE_NAME is None:
    if WHOLE_PIPE:
        out_folder = 'tested_images_whole_pipe_'+str(UPSCALE_FACTOR)+os.sep
    else:
        out_folder = 'tested_images_'+str(UPSCALE_FACTOR)+os.sep
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    test_set = TestDatasetFromFolder(TEST_DIR, upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    for image_name, lr_image, lr_in_hr_scale in test_loader:
        start = time.clock()
        lr = Variable(lr_image, volatile=True)
        if TEST_MODE:
            lr = lr.cuda()
        sr_image = model(lr)
        elapsed = (time.clock() - start)
        print('cost = ' + str(elapsed) + 's')
        test_images = []
        test_images.extend(
            [display_transform()(lr_in_hr_scale.squeeze(0)), display_transform()(sr_image.data.cpu().squeeze(0))])
        test_images = torch.stack(test_images)
        test_images = torch.chunk(test_images, test_images.size(0) // 2)
        index = 1
        for image in test_images:
            out_path = out_folder+image_name[0].split(".")[0].split("/")[-1]+".jpg"
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path, padding=5)
            index += 1
        print(image_name[0])

elif TEST_DIR is None and IMAGE_NAME is not None:
    image = Image.open(IMAGE_NAME)
    image = image.convert("RGB")
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME.split(".")[0]+'.jpg')
else:
    print("TEST SET OR INDIVIDUAL IMAGE NOT BOTH")
    sys.exit()
