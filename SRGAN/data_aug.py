import numpy as np
import imgaug
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='Image Augmentation')
parser.add_argument('--set_path', type=str, help='set of images')
parser.add_argument('--output_path', type=str, help='data augmentation folder')

opt = parser.parse_args()

# SET_PATH = opt.set_path
# OUTPUT_PATH = opt.output_path

SET_PATH = '/home/manuelrey/ESA/Dataset/Step2-SuperresolutionWhale/converted_jpg'
OUTPUT_PATH = '/home/manuelrey/ESA/Dataset/Step2-SuperresolutionWhale/data_aug'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# Image Augmentation
# Right/Left flip 50% of the time
augmentation = imgaug.augmenters.Sometimes(0.833, imgaug.augmenters.Sequential([
    imgaug.augmenters.Fliplr(0.5), # horizontal flips
    # imgaug.augmenters.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    imgaug.augmenters.Sometimes(0.5,
                                imgaug.augmenters.GaussianBlur(sigma=(0, 0.5))
                                ),
    # Strengthen or weaken the contrast in each image.
    imgaug.augmenters.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    imgaug.augmenters.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    imgaug.augmenters.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True))# apply augmenters in random order

# image_filenames = [os.path.join(SET_PATH, x) for x in os.listdir(SET_PATH) if is_image_file(x)]
image_filenames = []
for count, filename in enumerate(sorted(os.listdir(SET_PATH))):
    if is_image_file(filename):
        image_filenames.append(os.path.join(SET_PATH, filename))
images = []
image_names = []
for idx, img_file in enumerate(image_filenames):
    img = cv2.imread(img_file)
    img = cv2.resize(img, dsize=(128, 128))
    images.append(img)
    image_names.append(img_file)


aug_images = augmentation.augment_images(images)

for aug_img, img_name in zip(aug_images, image_names):
    name = str(img_name.split(".")[0].split("/")[-1])+"_data_aug.jpg"
    output_name = os.path.join(OUTPUT_PATH, name)
    cv2.imwrite(output_name, aug_img)




