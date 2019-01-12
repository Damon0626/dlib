# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-12 下午10:16
# @Email : wwymsn@163.com
# @Software: PyCharm

import cv2
import dlib
import numpy as np


# 将人脸转换为128D向量
facerec = dlib.face_recognition_model_v1('../dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()  # 检测人脸
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')  # 检测人脸轮廓

# 原始图片128D信息
img_origin = cv2.imread("../01-face_detector/messi5.jpg")
dets_origin = detector(img_origin, 1)  # 人脸区域
shape_origin = predictor(img_origin, dets_origin[0])
features_img_origin = facerec.compute_face_descriptor(img_origin, shape_origin)  # 生成128D
# print(len(features_img))

# 随便下载的messi图片测试
img_test = cv2.imread('ronaldo.jpg')
det_test = detector(img_test, 1)
shape_test = predictor(img_test, det_test[0])
features_img_test = facerec.compute_face_descriptor(img_test, shape_test)

# 计算欧式距离
f1 = np.array(features_img_origin)
f2 = np.array(features_img_test)

dist = np.sqrt(np.sum(np.square(f1 - f2)))
print(dist)  # 0.412 当＞0.4时认为相同

