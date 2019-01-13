# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-13 上午11:08
# @Email : wwymsn@163.com
# @Software: PyCharm


import cv2
import numpy as np
import dlib
from PIL import Image, ImageDraw, ImageFont


detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
cap.set(360, 480)

while True:
	flag, img = cap.read()
	rects = detector(img, 1)
	font1 = cv2.FONT_HERSHEY_SIMPLEX

	for k, d in enumerate(rects):
		cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
	cv2.putText(img, "Faces:"+str(len(rects)), (50, 80), font1, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

	pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # PIL和cv2的图像颜色码不一样
	draw = ImageDraw.Draw(pil_img)
	font = ImageFont.truetype('simsun.ttf', 20, encoding='utf-8')
	draw.text((d.left(), d.top()), "刘德华", (0, 255, 0), font=font)

	img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
	cv2.imshow('cap', img)
	if cv2.waitKey(1) == ord('q'):
		break
cv2.destroyAllWindows()
