import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from CNN_model import inference_few,loss_optimizer
import  Model_eval

Epochs = 20
Batch_Size = 32
Width = 64
Height = 64

train_imgs_path = 'train_img/'
train_labels_path = 'train_label/'
val_imgs_path = 'val_img/'
val_labels_path = 'val_label/'
Summary_Dir = 'logs/'
model_path = 'model/'

# 读取训练集
def read_img_label(imgs_path,labels_path):
    #存储图片名称和标签名称
    imgs = []
    labels = []

    imgs_name = os.listdir(imgs_path)
    for img_name in imgs_name:
        # 读取路径
        img_path = os.path.join(imgs_path, img_name)
        label_name = img_name[:-4] + '.txt'
        label_path = os.path.join(labels_path, label_name)

        img = Image.open(img_path)
        imgs.append(np.array(img))

        with open(label_path, 'r', encoding='UTF-8') as label:
            # 读取当前txt文件的所有内容
            label_lines = label.readlines()
        label = []
        # 将当前txt文件的每行切割
        for line in label_lines:
            one_line = line.strip().split('\n')
            one_line = float(one_line[0])
            label.extend([one_line])
        labels.append(label)
    return np.array(imgs,np.float32),np.array(labels,np.float32)

# 按批次取数据
def mini_batches(imgs, labels, batch_size):
    assert len(imgs) == len(labels)
    for start_idx in range(0, len(imgs) - batch_size + 1, batch_size):
        part_idx = slice(start_idx, start_idx + batch_size)
        # 程序执行到yield语句的时候，程序暂停，
        # 返回yield后面表达式的值，在下一次调用的时候，
        # 从yield语句暂停的地方继续执行，如此循环，直到函数执行完
        yield imgs[part_idx], labels[part_idx]

# 计算平均召回率 精度 IoU
def average_r_p_iou(y_label, pred, num):
    recall_, precision_, IoU_ = 0,0,0
    for idx in range(num):
        recall, precision, IoU = Model_eval.recall_precision_Iou(y_label[idx], pred[idx])
        recall_ += recall
        precision_ += precision
        IoU_ += IoU
    return recall_/num, precision_/num, IoU_/num

# 将训练或测试结果显示在图像中
def save_result_img(label,img_name):
    label = np.reshape(label, [1, 4, 2])
    label = label.astype(np.int32)
    img = np.zeros([64, 64])
    img = cv2.fillPoly(img, label, 255)
    cv2.imwrite(img_name, img)

if __name__ == '__main__':
    if not os.path.exists(Summary_Dir):
        os.mkdir(Summary_Dir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # 读取训练集 验证集
    x_train, y_train = read_img_label(train_imgs_path,train_labels_path)
    x_val, y_val = read_img_label(val_imgs_path,val_labels_path)
    print("训练集维度：",x_train.shape," 验证集维度：",x_val.shape)

    # 定义占位符变量
    with tf.name_scope('Input'):
        keep_prob = tf.placeholder(tf.float32)
        x_input = tf.placeholder(tf.float32, shape=[None, Height, Width, 3], name='x_input')
        y_input = tf.placeholder(tf.float32, shape=[None, 8], name='y_input')

    # 获得前向传播输出
    pred = inference_few(x_input,keep_prob)
    # 定义损失函数和优化算法
    smooth_loss, train_optim = loss_optimizer(pred, y_input)

    # 声明Saver类用于保存模型
    saver = tf.train.Saver()
    # 合并所有变量操作
    merged = tf.summary.merge_all()

    # 建立会话
    with tf.Session() as sess:
        # 初始化写日志的writer
        train_writer = tf.summary.FileWriter(Summary_Dir+'train/', sess.graph)
        val_writer = tf.summary.FileWriter(Summary_Dir+'val/')
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # epoch迭代训练
        for epoch in range(Epochs):

            train_loss, train_summary = 0, 0
            # batch迭代训练
            for x_input_batch, y_input_batch in mini_batches(x_train, y_train, Batch_Size):
                _, train_summary,  train_batch_loss, train_pred, train_label = \
                    sess.run([ train_optim, merged, smooth_loss, pred, y_input],
                                feed_dict={x_input: x_input_batch, y_input: y_input_batch, keep_prob: 0.5})
                train_loss += train_batch_loss

                if epoch == Epochs-1:
                    # print(train_label[0], train_pred[0])
                    print("train model eval:",Model_eval.recall_precision_Iou(train_label[0], train_pred[0]))
                    # print("train model eval:",average_r_p_iou(train_label, train_pred, Batch_Size))
                    # save_result_img(train_pred[0], "train_pred.jpg")
            # 将训练日志写入文件
            train_writer.add_summary(train_summary, epoch)

            # 验证过程
            val_summary,val_loss, val_pred = sess.run([merged,smooth_loss, pred],
                                   feed_dict={x_input: x_val, y_input: y_val, keep_prob: 1})
            if epoch ==Epochs-1:
                # print("val model eval:",Model_eval.recall_precision_Iou(y_val[0], val_pred[0]))
                print("val model eval:", average_r_p_iou(y_val, val_pred, len(x_val)))
                save_result_img(val_pred[10], "val_pred.jpg")
            # 将验证日志写入文件
            val_writer.add_summary(val_summary, epoch)

            # 打印信息
            print("epoch {}".format(epoch))
            print("train loss：", train_loss)
            print("val loss: %f" % (val_loss))
            print('*' * 50)

        # 保存模型 必须保存在文件夹下，不能保存在代码的根目录下
        saver.save(sess, model_path+'model.ckpt')
        # 关闭会话 关闭writer
        sess.close()
        train_writer.close()
        val_writer.close()
