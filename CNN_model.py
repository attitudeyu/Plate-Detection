import tensorflow as tf
import tensorflow.contrib.slim as slim

Learning_Rate = 0.001

# 卷积层
def conv_layer(input, name, kh, kw, num_out, dh, dw, set_padding='SAME'):
    # 转化输入为tensor类型
    input = tf.convert_to_tensor(input)
    # 获得输入特征图的深度
    num_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # 权重矩阵的xavier初始化
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, num_in, num_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # 卷积层
        conv = tf.nn.conv2d(input, # 卷积的输入图像[batch的图片数量, 图片高度, 图片宽度, 图像通道数]
                            kernel, # 卷积核[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
                            (1, dh, dw, 1), # 卷积时在图像每一维的步长
                            padding=set_padding) # 卷积方式
        # 偏差初始化
        bias_init_val = tf.constant(0.0, shape=[num_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        # 批标准化
        z = slim.batch_norm(inputs=z)
        # 计算激活结果
        activation = tf.nn.relu(z, name=scope)
        return activation

# 全连接层
def fc_layer(input_op, name, num_out):
    num_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[num_in, num_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[num_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer 先进行线性运算，再加上bias，最后非线性计算
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation

# 池化层
def pool_layer(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1], # 池化窗口的大小
                          strides=[1, dh, dw, 1], # 每一个维度上滑动的步长
                          padding='VALID',
                          name=name)

# 前向传播过程
def inference(input_op, keep_prob):
    conv1_1 = conv_layer(input_op, name="conv1_1", kh=3, kw=3, num_out=8, dh=1, dw=1)
    conv1_2 = conv_layer(conv1_1, name="conv1_2", kh=3, kw=3, num_out=8, dh=1, dw=1)
    pool1 = pool_layer(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    conv2_1 = conv_layer(pool1, name="conv2_1", kh=3, kw=3, num_out=16, dh=1, dw=1)
    conv2_2 = conv_layer(conv2_1, name="conv2_2", kh=3, kw=3, num_out=16, dh=1, dw=1)
    pool2 = pool_layer(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_layer(pool2, name="conv3_1", kh=3, kw=3, num_out=32, dh=1, dw=1)
    conv3_2 = conv_layer(conv3_1, name="conv3_2", kh=3, kw=3, num_out=32, dh=1, dw=1)
    pool3 = pool_layer(conv3_2, name="pool3", kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_layer(pool3, name="conv4_1", kh=3, kw=3, num_out=64, dh=1, dw=1)
    conv4_2 = conv_layer(conv4_1, name="conv4_2", kh=3, kw=3, num_out=64, dh=1, dw=1)
    pool4 = pool_layer(conv4_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    pool_shape = pool4.get_shape()
    flatten_shape = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pool4, [-1, flatten_shape], name="flatten")

    fc5 = fc_layer(flatten, name="fc5", num_out=128)
    fc5_drop = tf.nn.dropout(fc5, keep_prob, name="fc5_drop")

    fc6 = fc_layer(fc5_drop, name="fc6", num_out=64)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc5_drop")

    fc7 = fc_layer(fc6_drop, name="fc7", num_out=8)
    return fc7

def inference_few(input_op, keep_prob):
    conv1 = conv_layer(input_op, name="conv1_1", kh=3, kw=3, num_out=4, dh=1, dw=1)
    pool1 = pool_layer(conv1, name="pool1", kh=2, kw=2, dw=2, dh=2)

    pool_shape = pool1.get_shape()
    flatten_shape = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pool1, [-1, flatten_shape], name="flatten")

    fc1 = fc_layer(flatten, name="fc1", num_out=8)
    return fc1

# 定义损失函数、优化算法
def loss_optimizer(cnnum_out, y_train_node):
    # 均方差损失函数
    # mse_loss = tf.reduce_sum( tf.square(y_train_node - cnnum_out) )
    # mse_loss = tf.losses.mean_squared_error(y_train_node,cnnum_out)
    # Smooth L1损失函数
    smooth_loss = tf.losses.huber_loss(y_train_node,cnnum_out,delta=1.0)
    train_optim = tf.train.AdamOptimizer(Learning_Rate).minimize(smooth_loss)
    tf.summary.scalar('Loss',smooth_loss)
    return smooth_loss, train_optim