# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-13 下午8:38
# @Email : wwymsn@163.com
# @Software: PyCharm

'''
本例只做简单的测试，图片都为１张人脸．

人脸特征点对齐：
识别五官的位置，做好前期的准备工作，便于后续开展其他工作，如人脸替换＼表情识别等．
'''
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

ori_img = cv2.imread('alignment.jpg')
rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 更改颜色编码

dets = detector(rgb_img, 1)

faces = dlib.full_object_detections()
for det in dets:
	faces.append(predictor(rgb_img, det))

images = dlib.get_face_chips(rgb_img, faces, size=320)
#
# # 再将颜色转换回来
cv_rgb_img = np.array(images).astype(np.uint8)
cv_bgr_img = cv2.cvtColor(cv_rgb_img[0], cv2.COLOR_RGB2BGR)
cv2.imshow('alignment', cv_bgr_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 也可以用plt进行显示
# plt.imshow(cv_rgb_img[0])
# plt.show()