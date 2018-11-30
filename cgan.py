import tensorflow as tf
import numpy as np
from ops import *
from tensorflow.examples.tutorials.mnist import input_data

# 如果当前文件夹中没有mnist数据，则会自动下载
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 使用独热编码，也就是标签是一个二维逻辑矩阵
train = mnist.train  # train有两个属性：images: 55000*784 和 labels： 55000*10

global_step = tf.Variable(0, name='global_step', trainable=False)
y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')
images = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='real_images')
z = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='z')

# G是生成的假图片
with tf.variable_scope(tf.get_variable_scope()) as scope:
    G = generator(z, y)
    D, D_logits = discriminator(images, y)  # D、D_logits都是 BATCH_SIZE*1的
    D_, D_logits_ = discriminator(G, y, reuse=True)
    samples = sampler(z, y)

# 固定使用train.labels的前BATCH_SIZE个作为生成图片的标签，可以指定生成图片的数字
sample_labels = mnist.train.labels[0:BATCH_SIZE]

# 损失计算
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

# 生成器和判别器要更新的变量，用于 tf.train.Optimizer 的 var_list
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

# 由于使用了tf.layers.batch_normalization，需要添加下面的两行代码
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for epoch in range(25):
        for i in range(int(55000/BATCH_SIZE)):
            batch = mnist.train.next_batch(BATCH_SIZE)
            batch_images = np.array(batch[0]).reshape((-1, 28, 28, 1))
            batch_labels = batch[1]
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            sess.run([d_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
            sess.run([g_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
            sess.run([g_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
            if i % 100 == 0:
                errD = d_loss.eval(feed_dict={images: batch_images, y: batch_labels, z: batch_z})
                errG = g_loss.eval({z: batch_z, y: batch_labels})
                print("epoch:[%d], i:[%d]  d_loss: %.8f, g_loss: %.8f" % (epoch, i, errD, errG))
            # 在训练过程中得到生成器生成的假的图片并保存
            if i % 100 == 1:
                sample = sess.run(samples, feed_dict={z: batch_z, y: sample_labels})
                samples_path = './pics/'
                save_images(sample, [8, 8], samples_path + 'epoch_%d_i_%d.png' % (epoch, i))
                print('save image')
            # 定期保存模型
            # if i == (int(55000/BATCH_SIZE)-1):
            #     checkpoint_path = os.path.join('./check_point/DCGAN_model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=i+1)
            #     print('save check_point')

