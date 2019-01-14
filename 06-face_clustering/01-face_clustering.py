# -*-coding:utf-8-*-
# @Author: Damon0626
# @Time  : 19-1-14 下午9:24
# @Email : wwymsn@163.com
# @Software: PyCharm


import dlib
import cv2
import os
import glob

detector = dlib.get_frontal_face_detector()
shape_detector = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("../dlib_face_recognition_resnet_model_v1.dat")

description = []
images = []

# 读取friends目录下的所有jpg文件
for im in glob.glob(os.path.join("friends", "*.jpg")):
	print("Processing file:{}".format(im))
	img = cv2.imread(im)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# 检测人脸数
	dets = detector(img_rgb, 1)
	print("Number of faces detected{}".format(len(dets)))

	# 遍历所有人脸获得所有的128D特征
	for index, face in enumerate(dets):
		shape = shape_detector(img_rgb, face)
		shape_description = face_recognizer.compute_face_descriptor(img_rgb, shape)

		description.append(shape_description)
		images.append((img_rgb, shape))  # 存储图像和68点特征

labels = dlib.chinese_whispers_clustering(description, 0.5)
print("labes:{}".format(labels))
num_classes = len(set(labels))
print("Numbers of clusters:{}".format(num_classes))

face_dict = {}
for i in range(num_classes):
	face_dict[i] = []

for i in range(len(labels)):
	face_dict[labels[i]].append(images[i])


for key in face_dict.keys():  # key为分类，value为图像和其五官特征;
	file_dir = os.path.join("./cluster", str(key))
	if not os.path.isdir(file_dir):
		os.makedirs(file_dir)

	for i, (image, shape) in enumerate(face_dict[key]):
		file_path = os.path.join(file_dir, "face_"+str(i))
		print(file_path)
		dlib.save_face_chip(image, shape, file_path, size=150, padding=0.25)  # 150*150