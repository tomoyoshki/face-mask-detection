import cv2
import time
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np

model = tf.keras.models.load_model('saved_model/my_model')

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

def harr_rec():
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
		mask = mask_recognize(frame)
		#frame = frame[0:720, 320: 960, :]
		#frame = cv2.resize(frame, (128, 128))
		cv2.rectangle(frame, (320,2), (960,718), color=(255, 255, 255), thickness=2)
		print(mask)
		for x, y, width, height in faces:
			# 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
			if mask == 2:
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
			else:
				cv2.rectangle(frame, (x, y), (x + width, y + height), color=(0, 0, 255), thickness=2)
		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break
	vc.release()

def mask_recognize(img):
	frame = img.copy()
	frame = frame[0:720, 320: 960, :]
	frame = cv2.resize(frame, (128, 128))
	frame = np.array([frame])
	predict_x = model.predict(frame)
	classes_x = np.argmax(predict_x,axis=1)
	return classes_x
	
if __name__ == "__main__":
	#img = cv2.cvtColor(cv2.imread("img_with_face_mask.png"), cv2.COLOR_BGR2RGB)
	#img = np.array([img])
	#print(img.shape)
	harr_rec()
	

	
