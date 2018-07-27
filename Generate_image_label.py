import cv2
import numpy as np
import os

Width = 64
Height = 64
Imgs_num = 50

def generate_labels(Imgs_num):
    labels = []
    for num in range(Imgs_num):
        # 随机生成坐标
        x1 = np.random.randint(8,18)
        labels.extend([x1])
        y1 = np.random.randint(21,31)
        labels.extend([y1])
        x2 = np.random.randint(46,56)
        labels.extend([x2])
        y2 = np.random.randint(21,31)
        labels.extend([y2])
        x3 = np.random.randint(46,56)
        labels.extend([x3])
        y3 = np.random.randint(33,43)
        labels.extend([y3])
        x4 = np.random.randint(8,18)
        labels.extend([x4])
        y4 = np.random.randint(33,43)
        labels.extend([y4])
    return np.array(labels).reshape((Imgs_num,4,2))

def generate_imgs(labels, object_imgs_path, object_labels_path):
    imgs_num = len(labels)
    for idx in range(imgs_num):
        # 生成图片数组
        img = np.zeros((Height, Width, 3))
        fill_coord = labels[idx]
        # 填充图片
        fill_img = cv2.fillPoly(img, [fill_coord], (255,0,0))
        img_name = os.path.join(object_imgs_path, str(idx)+'.jpg')
        label_name = os.path.join(object_labels_path, str(idx)+'.txt')
        # 保存图片 坐标
        cv2.imwrite(img_name, fill_img)
        np.savetxt(label_name, fill_coord.reshape((8,1)))

if __name__=='__main__':
    object_imgs_path = "train_img/"
    object_labels_path = "train_label/"

    if not os.path.exists(object_imgs_path):
        os.mkdir(object_imgs_path)
    if not os.path.exists(object_labels_path):
        os.mkdir(object_labels_path)

    # 设置随机种子，固定随机数
    np.random.seed(1)
    labels = generate_labels(Imgs_num)
    generate_imgs(labels, object_imgs_path,object_labels_path)
