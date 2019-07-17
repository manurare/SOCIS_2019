# Pixel Recursive Super Resolution

TensorFlow implementation of [Pixel Recursive Super Resolution](https://arxiv.org/abs/1702.00783). This implementation contains:

![model](./assets/model.png)

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
	$ python test.py --use_gpu=False

## Samples

Training after 30000 iteration.

![sample.png](./assets/sample.png)


## Training details

cross entropy loss:

![curve.png](./assets/curve.png)


## Author

nilboy / [@nilboy](https://github.com/nilboy)
