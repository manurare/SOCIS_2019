from utils import *
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as inch
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from data import DataSet
from net import Net
import sys

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "batch_size")
flags.DEFINE_integer("device_id", 0, "gpu device id")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
conf = flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)

test_set = DataSet('data/test_whale.txt', 1, mode='test')
hr_images = test_set.get_sample[0]
lr_images = test_set.get_sample[1]

device_str = '/gpu:' + str(conf.device_id)
with tf.device(device_str):
    prsr_net = Net(hr_images, lr_images,'prsr')

saver = tf.train.Saver()

# new_saver = tf.train.import_meta_graph('models/final.ckpt-19.meta')
# saver.restore(sess, 'models/final.ckpt-19')
saver.restore(sess, 'models/final_256_128.ckpt-49')

all_vars_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)
for var in all_vars_new:
    print(var)
    # print(sess.run(var))

sess.run(test_set.iterator.initializer)
sess.run([lr_images, hr_images])
# hr_to_feed, lr_to_feed = sess.run([hr_images, lr_images])
# loss = sess.run(prsr_net.loss, feed_dict={prsr_net.train: False})
# print(loss)
#
mu = 1.1
it = 0
while True:
    try:
        print(hr_images.shape[1])
        print(lr_images.shape[1])
        c_logits = prsr_net.conditioning_logits
        p_logits = prsr_net.prior_logits
        lr_imgs = lr_images
        hr_imgs = hr_images
        np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
        hr_size = hr_images.shape[1]
        gen_hr_imgs = np.zeros((conf.batch_size, hr_size, hr_size, 3), dtype=np.float32)
        np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, prsr_net.train: False})

        for i in range(hr_size):
            print(i)
            for j in range(hr_size):
                for c in range(3):
                    np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
                    new_pixel = logits_2_pixel_value(
                        np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256], mu=mu)
                    gen_hr_imgs[:, i, j, c] = new_pixel

        name_to_save = str(it)+'_'+str(hr_images.shape[1])+'_'+str(lr_images.shape[1])+'.jpg'
        save_samples(np_lr_imgs, conf.samples_dir + '/lr_' + name_to_save)
        save_samples(np_hr_imgs, conf.samples_dir + '/hr_' + name_to_save)
        save_samples(gen_hr_imgs, conf.samples_dir + '/generate_' + name_to_save)
        it = it + 1
    except tf.errors.OutOfRangeError:
        print("END OF DATA")
        sess.close()
        sys.exit()


# def diff_letters(a,b):
#     u = zip(a, b)
#     d = dict(u)
#
#     x = 0
#     for i, j in d.items():
#         if i != j:
#             x = x+1
#     return x


# flags = tf.app.flags
# flags.DEFINE_integer("batch_size", 1, "batch_size")
# flags.DEFINE_integer("device_id", 1, "gpu device id")
# flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
# conf = flags.FLAGS
#
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = False
# sess = tf.Session(config=config)
#
# new_saver = tf.train.import_meta_graph('models/final.ckpt-19.meta')
# new_saver.restore(sess, 'models/final.ckpt-19')
#
# hr_images_ph = tf.placeholder(tf.float32,[conf.batch_size,256,256,3])
# lr_images_ph = tf.placeholder(tf.float32,[conf.batch_size,64,64,3])
# prsr_net = Net(hr_images_ph, lr_images_ph,'prsr')
# test_set = DataSet('data/test_whale.txt', 1, mode='test')
# hr_images = test_set.get_sample[0]
# lr_images = test_set.get_sample[1]
#
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)
# all_vars_new = tf.global_variables()
#
# for var in all_vars_new:
#     for check in all_vars_new:
#         var_split = var.name.split("/")
#         check_split = check.name.split("/")
#         first_var = "".join(var_split[:-1])
#         first_check = "".join(check_split[:-1])
#         if first_var == first_check:
#             diff = diff_letters(str(var_split[-1]), str(check_split[-1]))
#             if diff == 2:
#                 print(var)
#                 print(check)
#                 assign_op = tf.assign(check, var)
#                 sess.run(assign_op)
#                 break
#
# sess.run(test_set.iterator.initializer)
# hr_to_feed, lr_to_feed = sess.run([hr_images, lr_images])
# loss = sess.run(prsr_net.loss, feed_dict={prsr_net.train: False, hr_images_ph: hr_to_feed, lr_images_ph: lr_to_feed})
# print(loss)
#
#
# mu = 1.1
# c_logits = prsr_net.conditioning_logits
# p_logits = prsr_net.prior_logits
# lr_imgs = lr_images
# hr_imgs = hr_images
# np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
# hr_size = hr_images.shape[1]
# gen_hr_imgs = np.zeros((conf.batch_size, hr_size, hr_size, 3), dtype=np.float32)
# # gen_hr_imgs = np_hr_imgs
# # gen_hr_imgs[:,16:,16:,:] = 0.0
# np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, hr_imgs: np_hr_imgs, prsr_net.train: False})
#
# for i in range(hr_size):
#     for j in range(hr_size):
#         for c in range(3):
#             np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
#             new_pixel = logits_2_pixel_value(
#                 np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256], mu=mu)
#             gen_hr_imgs[:, i, j, c] = new_pixel
#
# save_samples(np_lr_imgs, conf.samples_dir + '/lr_' + str(mu * 10) + '.jpg')
# save_samples(np_hr_imgs, conf.samples_dir + '/hr_' + str(mu * 10) + '.jpg')
# save_samples(gen_hr_imgs, conf.samples_dir + '/generate_' + str(mu * 10) + '.jpg')




# flags = tf.app.flags
# flags.DEFINE_integer("batch_size", 1, "batch_size")
# flags.DEFINE_integer("device_id", 1, "gpu device id")
# flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
# conf = flags.FLAGS
#
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = False
# sess = tf.Session(config=config)
#
# new_saver = tf.train.import_meta_graph('models/final.ckpt-19.meta')
# new_saver.restore(sess, 'models/final.ckpt-19')
# hr_images_ph = tf.placeholder(tf.float32,[conf.batch_size,256,256,3])
# lr_images_ph = tf.placeholder(tf.float32,[conf.batch_size,64,64,3])
# prsr_net = Net(hr_images_ph, lr_images_ph,'prsr')
# test_set = DataSet('data/test_whale.txt', 1, mode='test')
# hr_images = test_set.get_sample[0]
# lr_images = test_set.get_sample[1]
#
# all_vars_new = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# # sess.run(init_op)
# for var in all_vars_new:
#     print(var)
#
# # for var in all_vars_new:
# #     var_checkpoint = str(var.name.split(":")[0])
# #     if(len(var_checkpoint.split("_"))) > 1:
# #
# #     var_to_init = str(var.name)
# #     print(var_checkpoint)
# #     print(var_to_init)
# #     tf.train.init_from_checkpoint('models/final.ckpt-19', {var_checkpoint: var})
# #     # print(sess.run(var))
#
# for var in all_vars_new:
#     for check in all_vars_new:
#         var_split = var.name.split("/")
#         check_split = check.name.split("/")
#         first_var = "".join(var_split[:-1])
#         first_check = "".join(check_split[:-1])
#         if first_var == first_check:
#             diff = diff_letters(str(var_split[-1]), str(check_split[-1]))
#             if diff == 2:
#                 var_checkpoint = str(var.name.split(":")[0])
#                 print(var_checkpoint)
#                 print(check.name)
#                 tf.train.init_from_checkpoint('models/final.ckpt-19', {var_checkpoint: str(check.name)})
#                 print(sess.run(check))
#                 break
#
# sess.run(test_set.iterator.initializer)
# hr_to_feed, lr_to_feed = sess.run([hr_images, lr_images])
# loss = sess.run(prsr_net.loss, feed_dict={prsr_net.train: False, hr_images_ph: hr_to_feed, lr_images_ph: lr_to_feed})
# print(loss)
#
#
# mu = 1.1
# c_logits = prsr_net.conditioning_logits
# p_logits = prsr_net.prior_logits
# lr_imgs = lr_images
# hr_imgs = hr_images
# np_hr_imgs, np_lr_imgs = sess.run([hr_imgs, lr_imgs])
# hr_size = hr_images.shape[1]
# gen_hr_imgs = np.zeros((conf.batch_size, hr_size, hr_size, 3), dtype=np.float32)
# # gen_hr_imgs = np_hr_imgs
# # gen_hr_imgs[:,16:,16:,:] = 0.0
# np_c_logits = sess.run(c_logits, feed_dict={lr_imgs: np_lr_imgs, hr_imgs: np_hr_imgs, prsr_net.train: False})
#
# for i in range(hr_size):
#     for j in range(hr_size):
#         for c in range(3):
#             np_p_logits = sess.run(p_logits, feed_dict={hr_imgs: gen_hr_imgs})
#             new_pixel = logits_2_pixel_value(
#                 np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256], mu=mu)
#             gen_hr_imgs[:, i, j, c] = new_pixel
#
# save_samples(np_lr_imgs, conf.samples_dir + '/lr_' + str(mu * 10) + '.jpg')
# save_samples(np_hr_imgs, conf.samples_dir + '/hr_' + str(mu * 10) + '.jpg')
# save_samples(gen_hr_imgs, conf.samples_dir + '/generate_' + str(mu * 10) + '.jpg')