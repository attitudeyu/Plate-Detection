import tensorflow as tf
from CNN_model import inference_few
import numpy as np
from PIL import  Image
import os

Width = 64
Height = 64
imgs_path = "test_img/"

def read_test_imgs(imgs_path):
    test_imgs_name = os.listdir(imgs_path)
    test_imgs = []
    for img_name in test_imgs_name:
        # 读取测试图片
        img_path = imgs_path+img_name
        img = Image.open(img_path)
        # 转化为tensor要求的浮点类型
        img = np.array(img, np.float32)
        # 数据集扩张维度
        img = img[np.newaxis,:,:,:]
        test_imgs.append(img)
    return np.array(test_imgs), test_imgs_name

# 保存预测坐标
test_label_path = "test_predict_label/"
if not os.path.exists(test_label_path):
    os.mkdir(test_label_path)

# 定义占位符变量
with tf.name_scope('Input'):
    keep_prob = tf.placeholder(tf.float32)
    x_input = tf.placeholder(tf.float32, shape=[1, Height, Width, 3], name='x_input')

# 获得前向传播输出
pred = inference_few(x_input,keep_prob)

saver = tf.train.Saver()

test_imgs , test_imgs_name = read_test_imgs(imgs_path)

with tf.Session() as sess:
    # 读入权重
    saver.restore(sess, "model/model.ckpt")
    for idx in range(len(test_imgs)):
        test_label = sess.run([pred],feed_dict={x_input: test_imgs[idx], keep_prob: 1})
        test_label = test_label[0][0]
        print("前向传播结果：",test_label)
        test_label_name = test_label_path+test_imgs_name[idx][:-4]+'.txt'
        np.savetxt(test_label_name,test_label)







