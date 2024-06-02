import cv2 as cv
import dlib

# 얼굴 탐지기 초기화
detector = dlib.get_frontal_face_detector()

# 웹캠 열기
cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if not ret:
		break
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = detector(gray)
	for face in faces:
		x, y, w, h = face.left(), face.top(), face.width(), face.height()
		cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
	cv.imshow('Face Detection', frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()