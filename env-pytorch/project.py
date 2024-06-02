import cv2
import dlib
import numpy as np

# Load dlib's face detector and the shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load the image
image_path = 'res/2024020811184781446_1.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Function to add a censorship bar over the eyes
def add_censorship_bar(image, landmarks):
	left_eye_start = 36
	left_eye_end = 41
	right_eye_start = 42
	right_eye_end = 47

	left_eye_points = [landmarks.part(i) for i in range(left_eye_start, left_eye_end + 1)]
	right_eye_points = [landmarks.part(i) for i in range(right_eye_start, right_eye_end + 1)]

	left_eye_center = np.mean([(p.x, p.y) for p in left_eye_points], axis=0).astype(int)
	right_eye_center = np.mean([(p.x, p.y) for p in right_eye_points], axis=0).astype(int)

	bar_thickness = 20
	bar_height = 10
	bar_width = 2 * bar_thickness

	# Add black bar over the left eye
	cv2.rectangle(image, (left_eye_center[0] - bar_thickness, left_eye_center[1] - bar_height), 
						 (left_eye_center[0] + bar_thickness, left_eye_center[1] + bar_height), (0, 0, 0), -1)
	# Add black bar over the right eye
	cv2.rectangle(image, (right_eye_center[0] - bar_thickness, right_eye_center[1] - bar_height), 
						 (right_eye_center[0] + bar_thickness, right_eye_center[1] + bar_height), (0, 0, 0), -1)

# Loop through each face detected
for face in faces:
	landmarks = predictor(gray, face)
	add_censorship_bar(image, landmarks)
	x, y, w, h = face.left(), face.top(), face.width(), face.height()
	cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Save and display the image
output_path = 'res/output_image.png'
cv2.imwrite(output_path, image)
cv2.imshow("Image with Censorship Bars", image)
cv2.waitKey(0)
cv2.destroyAllWindows()