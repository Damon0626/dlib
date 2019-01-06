# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-6 下午9:55
# @Email : wwymsn@163.com
# @Software: PyCharm

import sys
import dlib
import cv2


detecctor = dlib.get_frontal_face_detector()

img = cv2.imread('friends.jpg')
faces = detecctor(img, 1)
print("人脸数/ Faces in all:", len(faces))
# print(faces)

for i, d in enumerate(faces):
	# print(i, d)  # 0 [(581, 80), (689, 187)]  ((左右)，(上下))
	print("第", i+1, "个人脸的矩形框坐标是：", "left:", d.left(), "right:", d.right(), "top:", d.top(), "bottom:", d.bottom())
	cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)
