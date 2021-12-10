import cv2
import time
import tensorflow as tf
import pandas as pd
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow.keras.layers as tfl


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


model = tf.keras.models.load_model('saved_model/CNN_model')

def detect(image):
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	# 检测图像中的所有人脸
	faces = face_cascade.detectMultiScale(image_gray)
	#print(f"{len(faces)} faces detected in the image.")
	return faces
	
def rtsp():
	cv2.namedWindow("preview")
	rtsp_streaming = "rtmp://172.22.146.248/live/charles"
	for i in range(5):
		try:
			cap = cv2.VideoCapture(rtsp_streaming)
			break
		except:
			continue
	ret,frame = cap.read()
	while ret:
			ret,frame = cap.read()
			#cv2.imshow("frame",frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			faces = detect(frame)
			for x, y, width, height in faces:
				# 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
			cv2.imshow("preview", frame)
			#rval, frame = vc.read()
			key = cv2.waitKey(20)
			if key == 27: # exit on ESC
				break
	cv2.destroyAllWindows()
	cap.release()
	cv2.destroyWindow("preview")

def face_rec():
	detector = MTCNN()
	#Load a videopip TensorFlow
	video_capture = cv2.VideoCapture(0)
	 
	while (True):
		ret, frame = video_capture.read()
		frame = cv2.resize(frame, (600, 400))
		boxes = detector.detect_faces(frame)
		if boxes:
			box = boxes[0]['box']
			conf = boxes[0]['confidence']
			x, y, w, h = box[0], box[1], box[2], box[3]
			if conf > 0.5:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break
	video_capture.release()
	cv2.destroyAllWindows()

def harr_rec(model):
	#print(model.summary())
	cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
	vc = cv2.VideoCapture(0)

	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		#time.sleep( 0.2 )
		#print(frame.shape)
		faces = detect(frame)
		#mask = mask_recognize(frame)

		#frame = frame[0:720, 320: 960, :]
		#frame = cv2.resize(frame, (128, 128))
		cv2.rectangle(frame, (320,2), (960,718), color=(255, 255, 255), thickness=2)

		for x, y, width, height in faces:

			img= frame[y:y + height,x:x + width, :]
			if img.shape[0] == 0 or img.shape[1] == 0 or img.shape[2] == 0:
				continue
			print(img.shape)
			mask = mask_recognize(img, model)

			print(mask)
			print(x,y,width, height)
			# 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
			if mask == 1:
				# green
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255,0), thickness=2)
			elif mask == 2:
				#yellow
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255, 255), thickness=2)
			else:
				#red
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break
	vc.release()

def mask_recognize(img, model):


	frame = img.copy()

	#frame = frame[0:720, 320: 960, :]
	frame = cv2.resize(frame, (128, 128))
	cv2.imwrite("test.png", frame)
	plt.imshow(frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	frame = np.array([frame])

	predict_x = model.predict(frame)

	print(predict_x)
	classes_x = np.argmax(predict_x,axis=1)
	return classes_x


def train():
	imdir0 = '/Users/gary/Desktop/Dataset/without_mask/'
	imdir1 = '/Users/gary/Desktop/Dataset/with_mask/'
	imdir2 = '/Users/gary/Desktop/Dataset/mask_weared_incorrect/'

	ext = ['png']  # Add image formats here

	for i in range(3):
		files = []
		if (i == 0):
			[files.extend(glob.glob((imdir0) + '*.' + e)) for e in ext]
			without_mask_img = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
		if (i == 1):
			[files.extend(glob.glob((imdir1) + '*.' + e)) for e in ext]
			with_mask_img = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
		if (i == 2):
			[files.extend(glob.glob((imdir2) + '*.' + e)) for e in ext]
			mask_weared_incorrect_img = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]

	without_mask_img = np.array(without_mask_img)  # shape = (2994, 128, 128, 3) for all of the categories
	with_mask_img = np.array(with_mask_img)
	mask_weared_incorrect_img = np.array(mask_weared_incorrect_img)

	complete_data_set = np.concatenate((without_mask_img, with_mask_img, mask_weared_incorrect_img))
	# shape = (8982, 128, 128, 3)
	# print(complete_data_set.shape)
	labels = []
	for i in range(3):
		for j in range(2994):
			labels.append(i)

	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)

	X_train, X_test, y_train, y_test = train_test_split(complete_data_set, labels, test_size=0.3, random_state=42)
	model = tf.keras.Sequential([
		tfl.ZeroPadding2D(padding=3, input_shape=(128, 128, 3)),  # input image is 128*128*3
		tfl.Conv2D(32, 7, 1),
		tfl.BatchNormalization(3),
		tfl.ReLU(),
		tfl.MaxPooling2D(),
		tfl.Flatten(),
		tfl.Dense(3, 'softmax')
	])

	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=15, verbose=1)

	model.save('saved_model/CNN_model')
	return model


if __name__ == "__main__":

	#img = cv2.cvtColor(cv2.imread("img_with_face_mask.png"), cv2.COLOR_BGR2RGB)
	#img = np.array([img])
	#print(img.shape)
	harr_rec(model)



	

	
