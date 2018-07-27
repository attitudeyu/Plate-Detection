import os
import numpy as np
import cv2

train_imgs="train_img/"
train_labels='train_label/'
aug_rotate_imgs = "aug_rotate_img/"
aug_rotate_labels = "aug_rotate_label/"
aug_trans_imgs = "aug_trans_img/"
aug_trans_labels = "aug_trans_label/"
if not os.path.exists(aug_rotate_imgs):
    os.mkdir(aug_rotate_imgs)
if not os.path.exists(aug_rotate_labels):
    os.mkdir(aug_rotate_labels)
if not os.path.exists(aug_trans_imgs):
    os.mkdir(aug_trans_imgs)
if not os.path.exists(aug_trans_labels):
    os.mkdir(aug_trans_labels)

Rotate_Select_Imgs = 50
Angles = 4
Trans_Select_Imgs = 50

def draw_label(img, label):
    color = (0,0,255)
    label = label.astype(np.int32)
    cv2.line(img, tuple(label[0]), tuple(label[1]), color)
    cv2.line(img, tuple(label[1]), tuple(label[2]), color)
    cv2.line(img, tuple(label[2]), tuple(label[3]), color)
    cv2.line(img, tuple(label[3]), tuple(label[0]), color)
    cv2.imshow("img",img)
    cv2.waitKey(0)

def read_imgs_labels(imgs_path, labels_path):
    imgs_name = os.listdir(imgs_path)
    imgs = []
    labels = []
    for img_name in imgs_name:
        img_path = os.path.join(imgs_path, img_name)
        label_name = img_name[:-4]+'.txt'
        label_path = os.path.join(labels_path, label_name)

        img = cv2.imread(img_path)
        # print("the type of readed img:",type(img))
        imgs.append(img)

        with open(label_path, 'r', encoding='UTF-8') as label:
            label_lines = label.readlines()
        label = []
        for line in label_lines:
            one_line = line.strip().split('\n')[0]
            label.extend([float(one_line)])
        labels.append(label)
    return np.array(imgs), np.array(labels)

# 旋转
def rotate_img_label(img, angle, label):
    (height, width) = img.shape[:2]
    center = (height//2, width//2)
    # 计算旋转矩阵  图片中心，旋转角度，图像旋转后的大小
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转图像
    rotate_img = cv2.warpAffine(img, matrix, (width,height))
    # 旋转坐标
    matrix = np.row_stack((matrix, np.array([0,0,1])))
    label = np.row_stack((label, np.array([1,1,1,1])))
    rotate_label = np.dot(matrix, label)
    rotate_label = rotate_label[:2].transpose()
    return rotate_img, rotate_label

def get_rotate_imgs_lalbes(imgs, labels):
    imgs_num = len(imgs)
    np.random.seed(1)
    # 随机选取图片，其再随机产生随机角度旋转
    random_img = np.random.randint(0, imgs_num, Rotate_Select_Imgs)
    for idx_img in random_img:
        random_angle = np.random.randint(-180, 180, Angles)
        for angle in random_angle:
            # 调整为仿射矩阵要求的维度
            label = labels[idx_img].reshape((4, 2)).transpose()
            rotate_img, rotate_label = rotate_img_label(imgs[idx_img], angle, label)
            # 调整为神经网络要求的维度
            rotate_label = rotate_label.reshape((8, 1))
            # 保存旋转图片和旋转坐标
            rotate_img_name = str(idx_img) + str('_') + str(angle) + '.jpg'
            rotate_label_name = str(idx_img) + str('_') + str(angle) + '.txt'
            rotate_img_path = os.path.join(aug_rotate_imgs, rotate_img_name)
            rotate_label_path = os.path.join(aug_rotate_labels, rotate_label_name)
            cv2.imwrite(rotate_img_path, rotate_img)
            np.savetxt(rotate_label_path, rotate_label)

# 平移
def translate_img_label(img,x_shift, y_shift, label):
    (height, width) = img.shape[:2]
    # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
    matrix = np.float32([[1,0,x_shift],[0,1,y_shift]])
    # 平移图像
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    # 平移坐标
    for idx in range(8):
        if idx%2==0:
            label[idx] += x_shift
        else:
            label[idx] += y_shift
    return trans_img, label.reshape((4,2))

def get_trans_imgs_labels(imgs, labels):
    imgs_num = imgs.shape[0]
    np.random.seed(2)
    random_imgs = np.random.randint(0, imgs_num, Trans_Select_Imgs)
    for img_idx in random_imgs:
        # 获得随机平移坐标
        x_shift = np.random.randint(-8, 8, 1)
        y_shift = np.random.randint(-8, 8, 1)
        trans_img, trans_label = translate_img_label(imgs[img_idx], x_shift, y_shift, labels[img_idx])
        # 调整为神经网络要求的维度
        trans_label = trans_label.reshape((8,1))
        # 保存平移图片和平移坐标
        trans_img_name = str(img_idx)+'_'+str(x_shift[0])+'_'+str(y_shift[0])+'.jpg'
        trans_label_name = str(img_idx)+'_'+str(x_shift[0])+'_'+str(y_shift[0])+'.txt'
        trans_img_path = os.path.join(aug_trans_imgs, trans_img_name)
        trans_label_path = os.path.join(aug_trans_labels, trans_label_name)
        cv2.imwrite(trans_img_path, trans_img)
        np.savetxt(trans_label_path, trans_label)


if __name__=="__main__":
    imgs, labels = read_imgs_labels(train_imgs, train_labels)

    # 数据增强
    get_rotate_imgs_lalbes(imgs, labels)
    get_trans_imgs_labels(imgs, labels)


