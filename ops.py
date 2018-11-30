import tensorflow as tf
import scipy.misc
import numpy as np

BATCH_SIZE = 100


def weight_variable(shape, name, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(
                              stddev=stddev, dtype=dtype))
    return var


def bias_variable(shape, name, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


def conv2d(x, output_channels, name, k_h=5, k_w=5):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        w = weight_variable(shape=[k_h, k_w, x_shape[-1], output_channels], name='weights')
        b = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME') + b
        return conv


def deconv2d(x, output_shape, name, k_h=5, k_w=5):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # 注意这里的W的格式为 [height, width, output_channels, in_channels]
        w = weight_variable([k_h, k_w, output_shape[-1], x_shape[-1]], name='weights')
        bias = bias_variable([output_shape[-1]], name='biases')
        deconv = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 2, 2, 1], padding='SAME') + bias
        return deconv


def fully_connect(x, channels_out, name):
    shape = x.get_shape().as_list()
    channels_in = shape[1]
    with tf.variable_scope(name):
        weights = weight_variable([channels_in, channels_out], name='weights')
        biases = bias_variable([channels_out], name='biases')
        return tf.matmul(x, weights) + biases


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def conv_cond_concat(value, cond):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3)


def relu(value):
    return tf.nn.relu(value)


#  定义生成器，z:?*100, y:?*10
def generator(z, y, training=True):
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name="yb")  # y:?*1*1*10
    z = tf.concat([z, y], 1)  # z:?*110

    # 进过一个全连接、 batch_norm、和relu
    h1 = fully_connect(z, 1024, name='g_h1_fully_connect')
    h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=training, name='g_h1_batch_norm'))
    h1 = tf.concat([h1, y], 1)  # h1: ?*1034

    h2 = fully_connect(h1, 128 * 49, name='g_h2_fully_connect')
    h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=training, name='g_h2_batch_norm'))
    h2 = tf.reshape(h2, [BATCH_SIZE, 7, 7, 128])  # h2: ?*7*7*128
    h2 = conv_cond_concat(h2, yb)  # h2: ?*7*7*138

    h3 = deconv2d(h2, output_shape=[BATCH_SIZE, 14, 14, 128], name='g_h3_deconv2d')
    h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=training, name='g_h3_batch_norm'))  # h3: ?*14*14*128
    h3 = conv_cond_concat(h3, yb)  # h3:?*14*14*138

    h4 = deconv2d(h3, output_shape=[BATCH_SIZE, 28, 28, 1], name='g_h4_deconv2d')
    h4 = tf.nn.sigmoid(h4)  # h4: ?*28*28*1
    return h4


def discriminator(image, y, reuse=False, training=True):
    # with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')  # BATCH_SIZE*1*1*10
    x = conv_cond_concat(image, yb)  # image: BATCH_SIZE*28*28*1 ,x: BATCH_SIZE*28*28*11

    h1 = conv2d(x, 11, name='d_h1_conv2d')
    h1 = lrelu(tf.layers.batch_normalization(h1, name='d_h1_batch_norm', training=training,
                                             reuse=reuse))  # h1: BATCH_SIZE*14*14*11
    h1 = conv_cond_concat(h1, yb)  # h1: BATCH_SIZE*14*14*21

    h2 = conv2d(h1, 74, name='d_h2_conv2d')
    h2 = lrelu(
        tf.layers.batch_normalization(h2, name='d_h2_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*7*7*74
    h2 = tf.reshape(h2, [BATCH_SIZE, -1])  # BATCH_SIZE*3626
    h2 = tf.concat([h2, y], 1)  # BATCH_SIZE*3636

    h3 = fully_connect(h2, 1024, name='d_h3_fully_connect')
    h3 = lrelu(
        tf.layers.batch_normalization(h3, name='d_h3_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*1024
    h3 = tf.concat([h3, y], 1)  # BATCH_SIZE*1034

    h4 = fully_connect(h3, 1, name='d_h4_fully_connect')  # BATCH_SIZE*1
    return tf.nn.sigmoid(h4), h4


def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)


def save_images(images, size, path):
    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    # 保存画布
    return scipy.misc.imsave(path, merge_img)
