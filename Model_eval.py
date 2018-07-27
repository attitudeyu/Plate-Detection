import  numpy as np
import cv2

Width = 64
Height = 64

# 获得标注坐标的图片数组和非0区域的像素个数
def coord_convert_arr(coord):
    img = np.zeros((Height, Width))
    fill_coord = np.array(coord, np.int32).reshape((4,2))
    new_img = cv2.fillPoly(img, [fill_coord], 1) #2121
    # 获得非0区域的像素个数
    non_zero = cv2.countNonZero(new_img)+1
    # 转化为单行数组
    arr = new_img.reshape((1,Height*Width))[0]
    return arr, non_zero

# 获得相交区域的像素个数
def inter_pixel_num(label_arr, pred_arr):
    count = 0
    for idx in range(Width*Height):
        if label_arr[idx]==1 and pred_arr[idx]==1:
            count +=1
    return count

# 根据像素个数计算召回 精度 IoU
def recall_precision_Iou(label, pred):
    label_arr, label_area = coord_convert_arr(label)
    pred_arr, pred_area = coord_convert_arr(pred)
    inter_area = inter_pixel_num(label_arr, pred_arr)

    recall = inter_area/float(label_area)
    precision = inter_area/float(pred_area)
    IoU= inter_area/float(label_area+pred_area-inter_area)
    return recall, precision, IoU