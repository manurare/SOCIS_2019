from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
from data import *
from net import *
from utils import *
import os
import time
import matplotlib.pyplot as plt

flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
  def __init__(self):
    self.device_id = conf.device_id
    self.train_dir = conf.train_dir
    self.samples_dir = conf.samples_dir
    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    if not os.path.exists(self.samples_dir):
      os.makedirs(self.samples_dir)
      #datasets params
    self.num_epoch = conf.num_epoch
    self.batch_size = conf.batch_size
    #optimizer parameter
    self.learning_rate = conf.learning_rate

    # dataset
    self.dataset = DataSet(conf.imgs_list_path, self.batch_size)
    if conf.use_gpu:
      device_str = '/gpu:' +  str(self.device_id)
    else:
      device_str = '/cpu:0'
    with tf.device(device_str):
      self.hr_images = self.dataset.get_sample[0]
      self.lr_images = self.dataset.get_sample[1]
      self.net = Net(self.hr_images, self.lr_images, 'prsr')
      print(self.hr_images.shape)
      print(self.lr_images.shape)

      # self.net = Net(self.dataset.hr_images, self.dataset.lr_images, 'prsr')
      #optimizer
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                 500000, 0.5, staircase=True)
      optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
      self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)

  def train(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    #saver.restore(sess, './models/model.ckpt-30000')
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

    num_epochs = conf.num_epoch

    for epoch in range(num_epochs):
      print("Epoch: ", epoch)
      num_batch = 0
      sess.run(self.dataset.iterator.initializer)
      while True:
        try:
          # hr_imgs, lr_imgs = sess.run([self.hr_images, self.lr_images])
          # enc_hr = tf.image.encode_png(hr_imgs[0])
          # fname = tf.constant('test/hr.png')
          # fwrite = tf.write_file(fname, enc_hr)
          # sess.run(fwrite)
          # enc_hr = tf.image.encode_png(lr_imgs[0])
          # fname = tf.constant('test/lr.png')
          # fwrite = tf.write_file(fname, enc_hr)
          # sess.run(fwrite)
          # sys.exit()

          num_batch += 1
          t1 = time.time()
          _, loss, hr_imgs, lr_imgs = sess.run([self.train_op, self.net.loss,
                                                self.hr_images, self.lr_images], feed_dict={self.net.train: True})
          t2 = time.time()
          print('batch %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                % (num_batch, loss, self.batch_size / (t2 - t1), (t2 - t1)))

          if epoch % 10 == 0 and epoch != 0:
            checkpoint_path = os.path.join(self.train_dir, 'model'+'_'+str(hr_imgs.shape[1])+'_'+str(lr_imgs.shape[1])+'.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)
          if epoch == num_epochs - 1:
            checkpoint_path = os.path.join(self.train_dir, 'final'+'_'+str(hr_imgs.shape[1])+'_'+str(lr_imgs.shape[1])+'.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch)

        except tf.errors.OutOfRangeError:
          sess.run(self.dataset.iterator.initializer)
          summary_str = sess.run(summary_op, feed_dict={self.net.train: True})
          summary_writer.add_summary(summary_str, epoch)
          # self.sample(sess, int(self.hr_images.shape[1]), mu=1.1, step=epoch)
          print("END OF DATA")
          break
    sess.close()

    # # Start input enqueue threads.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # iters = 0
    # try:
    #   while not coord.should_stop():
    #     # Run training steps or whatever
    #     t1 = time.time()
    #     _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.net.train: True})
    #     t2 = time.time()
    #     print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
    #     iters += 1
    #     # print(self.dataset.hr_images.shape)
    #     # print(self.dataset.lr_images.shape)
    #     if iters % 10 == 0:
    #       summary_str = sess.run(summary_op, feed_dict={self.net.train: True})
    #       summary_writer.add_summary(summary_str, iters)
    #     if iters % 1000 == 0:
    #       #self.sample(sess, mu=1.0, step=iters)
    #       self.sample(sess, int(self.dataset.hr_images.shape[1]), mu=1.1, step=iters)
    #       #self.sample(sess, mu=100, step=iters)
    #     if iters % 10000 == 0:
    #       checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
    #       saver.save(sess, checkpoint_path, global_step=iters)
    # except tf.errors.OutOfRangeError:
    #   checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
    #   saver.save(sess, checkpoint_path)
    #   print('Done training -- epoch limit reached')
    # finally:
    #   # When done, ask the threads to stop.
    #   coord.request_stop()
    #
    # # Wait for threads to finish.
    # coord.join(threads)
    # sess.close()