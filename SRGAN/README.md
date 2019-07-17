# SRGAN
Modification of this [repository](https://github.com/leftthomas/SRGAN.git) implementation.

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```
## Usage

### Train
```
python train.py

optional arguments:
--hr_size                     size of the images to be generated [default value is 256]
--upscale_factor              super resolution upscale factor [default value is 4](choices:[2, 4, 8])
--num_epochs                  train epoch number [default value is 100]
--batch_size                  
```

### Test Single Image
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.

### Test batch of images
```
python test_image.py

optional arguments:
--upscale_factor              super resolution upscale factor [default value is 4]
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
--test_set_dir                Path where the test set is located
```
The output super resolution images are saved on tested_images/.
