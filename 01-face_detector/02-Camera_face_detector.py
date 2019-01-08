# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-8 下午10:31
# @Email : wwymsn@163.com
# @Software: PyCharm


'''
＠根据单目标追踪的实例来尝试人脸跟踪

使用dlib.correlation_tracker实现目标跟踪基本分以下四步：
１－dlib.correlation_tracker()创建一个跟踪类；

２－start_track()中设置图片中的要跟踪物体的框；

３－update()实时跟踪下一帧；

４－get_position()得到跟踪到的目标的位置。

＠将首帧检测到的人脸识别出后，将位置定位为追踪目标的位置，进而进行追踪．效果不是特别好，移动的过程中追踪框会追丢，后续在更新．
'''

import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()  # 人脸识别
tracker = dlib.correlation_tracker()  # 追踪

face_area = None
track_window = None
drag_start = None

frame_num = 0
while True:
	ret, frame = cap.read()
	first_frame = frame
	image = first_frame.copy()

	if frame_num == 0:  # 第一帧获得检测到的人脸位置
		pos = detector(first_frame, 0)
		for i, d in enumerate(pos):
			track_window = [d.top(), d.bottom(), d.left(), d.right()]
		tracker.start_track(image, dlib.rectangle(track_window[2], track_window[0], track_window[3], track_window[1]))  # 将检测到的人脸定为追踪目标

	else:  # 其余帧，更新实时追踪
		tracker.update(image)
		frame_num += 1

	box_predict = tracker.get_position()  # 得到目标的位置
	cv2.rectangle(image, (int(box_predict.left()), int(box_predict.top())), (int(box_predict.right()), int(box_predict.bottom())), (255, 0, 0), 2)
	cv2.imshow('img', image)

	if cv2.waitKey(5) == ord('q'):
		break
	frame_num += 1

cv2.destroyAllWindows()