# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-6 下午10:25
# @Email : wwymsn@163.com
# @Software: PyCharm


import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

img = cv2.imread("../01-face_detector/messi5.jpg")

dets = detector(img, 1)
print("检测到的人脸数为：", len(dets))

for index, face in enumerate(dets):
	print("第", index+1, "个人脸的矩形框坐标是：", "left:", face.left(), "right:", face.right(), "top:", face.top(), "bottom:", face.bottom())

	shape = predictor(img, face)
	# print(shape)
	for index, part in enumerate(shape.parts()):
		print("Part{}:{}".format(index, part))
		part_pos = (part.x, part.y)
		cv2.circle(img, part_pos, 1, (255, 0, 0), 1)

	cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("img", img)
cv2.waitKey(0)
