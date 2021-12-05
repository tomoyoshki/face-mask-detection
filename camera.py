import cv2
import time

def detect(image):
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	# 检测图像中的所有人脸
	faces = face_cascade.detectMultiScale(image_gray)
	print(f"{len(faces)} faces detected in the image.")
	return faces
	

if __name__ == "__main__":
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)

	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		#time.sleep( 0.2 )
		# frame = frame[0:360, 0: 640, :]
		
		#print(frame.shape)
		faces = detect(frame)
		for x, y, width, height in faces:
			# 这里的color是 蓝 黄 红，与rgb相反，thickness设置宽度
			cv2.rectangle(frame, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break

	vc.release()
	cv2.destroyWindow("preview")