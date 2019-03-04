#https://www.youtube.com/watch?v=BICsN-4B6HY
#https://github.com/nimbus1212/driver_fatigue_detection
# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from pygame import mixer # Load the required library
import matplotlib.pyplot as plt
from skimage import exposure
import operator

mixer.init()
mixer.music.load('alarm.mp3')


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3
YAWN_AR_THRESH = 32

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# Se crea la funcion para visualizar la imagen
def imshow(img):
	fig, ax = plt.subplots(figsize=(7,7))
	ax.imshow(img,cmap=plt.cm.gray)
	ax.set_xticks([]),ax.set_yticks([])
	plt.show()

def get_landmarks(im):
	rects = detector(im, 1)

	if len(rects) > 1:
		return "error"
	if len(rects) == 0:
		return "error"
	return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
	im = im.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0, 0], point[0, 1])
		#cv2.putText(im, str(idx), pos,
		#			fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
		#			fontScale=0.25,
		#			color=(0, 0, 255))
		cv2.circle(im, pos, 1, color=(0, 255, 255))
	return im


def top_lip(landmarks):
	top_lip_pts = []
	for i in range(48, 54):
		top_lip_pts.append(landmarks[i])
	for i in range(60, 64):
		top_lip_pts.append(landmarks[i])
	top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
	top_lip_mean = np.mean(top_lip_pts, axis=0)
	return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
	bottom_lip_pts = []
	for i in range(65, 68):
		bottom_lip_pts.append(landmarks[i])
	for i in range(55, 59):
		bottom_lip_pts.append(landmarks[i])
	bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
	bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
	return int(bottom_lip_mean[:, 1])


def mouth_open(image):
	landmarks = get_landmarks(image)

	if landmarks == "error":
		return image, 0

	image_with_landmarks = annotate_landmarks(image, landmarks)
	top_lip_center = top_lip(landmarks)
	bottom_lip_center = bottom_lip(landmarks)
	lip_distance = abs(top_lip_center - bottom_lip_center)
	return image_with_landmarks, lip_distance



fileStream = True
time.sleep(1.0)

cap = cv2.VideoCapture(0)
# loop over frames from the video stream

calibra = False
eyes_calibrado = 0
open_eyes = {}
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# add the equalize in image
	gray = exposure.equalize_adapthist(gray, clip_limit=0.05)
	data = 255 * gray
	gray = data.astype(np.uint8)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
		frame = annotate_landmarks(frame, landmarks)
		shape = face_utils.shape_to_np(shape)



		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		ear_round = round(ear,2)
		print(ear_round)
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	if len(rects)>0 and eyes_calibrado > 0:

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear > EYE_AR_THRESH:
			COUNTER += 1
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if eyes_calibrado>0:
			cv2.putText(frame, "CALIBRE: {:.2f}".format(eyes_calibrado), (150, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
	cv2.imshow("Frame", frame)


	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# if the `c` key was pressed, break from the loop
	if key == ord("c"):
		if calibra:
			calibra = False
			sorted_x = sorted(open_eyes.items(), key=operator.itemgetter(1))
			#eyes_calibrado = round((sorted_x[0][0]+ sorted_x[len(sorted_x)-1][0])/2,2)
			l = list(open_eyes)
			l.sort(reverse=True)
			eyes_calibrado = round(l[0], 2)
			print(eyes_calibrado)
			EYE_AR_THRESH = eyes_calibrado
		else:
			calibra = True
			eyes_calibrado = 0
			open_eyes = {}
			TOTAL = 0

	if calibra:
		print("calibrando")
		if ear_round in open_eyes:
			open_eyes[ear_round] += 1
		else:
			open_eyes[ear_round] = 1

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()