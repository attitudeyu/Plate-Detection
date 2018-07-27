import os
import numpy as np
import cv2

def read_label(label_path):
    labels = []
    # 读取标签
    with open(label_path, 'r', encoding='UTF-8') as label:
        # 读取当前txt文件的所有内容
        label_lines = label.readlines()
    label = []
    # 将当前txt文件的每行切割
    for idx, line in enumerate(label_lines):
        one_line = line.strip().split('\n')
        one_line = float(one_line[0])
        label.extend([one_line])
    labels.append(label)
    return np.array(labels,np.float32).reshape((4,2))

def draw_box_label(img_path,label, save_img_path):
    img = cv2.imread(img_path)
    color = (0,0,255)
    #绘制顶点
    for idx, coord in enumerate(label):
        cv2.circle(img, (coord[0],coord[1]),3,color)
        cv2.putText(img, str(idx+1), (coord[0],coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    #绘制边框
    cv2.line(img, (label[0][0],label[0][1]), (label[1][0],label[1][1]), color)
    cv2.line(img, (label[1][0],label[1][1]), (label[2][0],label[2][1]), color)
    cv2.line(img, (label[2][0],label[2][1]), (label[3][0],label[3][1]), color)
    cv2.line(img, (label[3][0],label[3][1]), (label[0][0],label[0][1]), color)
    # 保存图片
    cv2.imwrite(save_img_path, img)
    # 显示图片
    # cv2.namedWindow("img")
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

if __name__ =='__main__':
    save_imgs_path = 'test_img_pred/'
    if not os.path.exists(save_imgs_path):
        os.mkdir(save_imgs_path)

    imgs_path = 'test_img/'
    labels_path = 'test_predict_label/'

    imgs_name = os.listdir(imgs_path)
    for img_name in imgs_name:
        img_path = os.path.join(imgs_path,img_name)
        label_name = img_name[:-4]+'.txt'
        label_path = os.path.join(labels_path,label_name)
        save_img_path = os.path.join(save_imgs_path, img_name)

        label = read_label(label_path)
        draw_box_label(img_path, label, save_img_path)
