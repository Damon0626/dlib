# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-8 下午10:20
# @Email : wwymsn@163.com
# @Software: PyCharm


import dlib
import cv2


cnn_face_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

img = cv2.imread('../01-face_detector/friends.jpg')

dets = cnn_face_detector(img, 1)
# print(dets)

for i, face in enumerate(dets):
	d = face.rect
	print("第", i + 1, "个人脸的矩形框坐标是：", "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
	cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

cv2.namedWindow('img')
cv2.imshow('img', img)
cv2.waitKey(0)