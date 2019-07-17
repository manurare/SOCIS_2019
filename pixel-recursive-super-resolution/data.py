from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt

class DataSet(object):
  def __init__(self, images_list_path, batch_size, mode='train'):
    # #filling the record_list
    # input_file = open(images_list_path, 'r')
    # self.record_list = []
    # for line in input_file:
    #   line = line.strip()
    #   self.record_list.append(line)
    # filename_queue = tf.train.string_input_producer(self.record_list, num_epochs=num_epoch)
    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(filename_queue)
    # image = tf.image.decode_jpeg(image_file, 3)
    #
    # # preprocess
    # # hr images size should be power of 2 and divisible by the size of lr images which will be power of 2 as well
    #
    # hr_image = tf.image.resize_images(image, [128, 128])
    # lr_image = tf.image.resize_images(image, [32, 32])
    # hr_image = tf.cast(hr_image, tf.float32)
    # lr_image = tf.cast(lr_image, tf.float32)
    # #
    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 400 * batch_size
    # self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity,
    #   min_after_dequeue=min_after_dequeue)

    # ##### PRUEBAS
    self.length_dataset = self.get_dataset_size(images_list_path)
    self.dataset = tf.data.TextLineDataset(images_list_path)
    self.dataset = self.dataset.shuffle(self.length_dataset)
    # dataset = dataset.map(self.data_augmentation)
    if mode=='train':
      self.dataset = self.dataset.map(self.resize_hr_lr_train)
    else:
      self.dataset = self.dataset.map(self.resize_hr_lr_test)
    self.dataset = self.dataset.batch(batch_size)
    self.iterator = tf.data.make_initializable_iterator(self.dataset)
    self.dataset = self.dataset.prefetch(1)
    self.get_sample = self.iterator.get_next()

  def resize_hr_lr_train(self, filename):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)
    hr_image = tf.image.resize_images(image, [256, 256])
    lr_image = tf.image.resize_images(image, [128, 128])
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    return hr_image, lr_image

  def resize_hr_lr_test(self, filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    hr_image = tf.image.resize_images(image, [256, 256])
    lr_image = tf.image.resize_images(image, [128, 128])
    hr_image = tf.cast(hr_image, tf.float32)
    lr_image = tf.cast(lr_image, tf.float32)
    return hr_image, lr_image

  def get_dataset_size(self, fname):
    with open(fname) as f:
      for i, l in enumerate(f):
        pass
    return i + 1
