# Pixel Recursive Super Resolution

Modification from the implementation of this [repository](https://github.com/nilboy/pixel-recursive-super-resolution.git)

## Requirements

- Python 2.7
- [Skimage](http://scikit-image.org/)
- [TensorFlow](https://www.tensorflow.org/) 1.0


## Usage
Then, create image_list file:

	$ python tools/create_img_lists.py --dataset=data/train_set --outfile=data/train.txt

To train model on gpu:

	$ python tools/train.py
	(or $ python tools/train.py --device_id=0)

To train model on cpu:

	$ python tools/train.py --use_gpu=False
	
To test images:

	$ python test.py

## Clarifications
Images need to be squared of size power of 2. 
