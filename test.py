from ops import generator, save_images
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

BATCH_SIZE = 100
checkpoint_dir = './check_point/'

# ----------
y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])
z = tf.placeholder(tf.float32, [BATCH_SIZE, 100])
G = generator(z, y)
# -----------
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 使用独热编码，也就是标签是一个二维逻辑矩阵
train = mnist.train
sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
sample_labels = train.labels[120: 120 + BATCH_SIZE]

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

images = sess.run(G, feed_dict={z: sample_z, y: sample_labels})
save_images(images, [8, 8], 'test.png')
sess.close()
