import dlib
import cv2
from math import hypot

predictorModel = "shape_predictor_model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictorModel)

# Initialize variables for smoothing
previous_dist = None
alpha = 0.5  # Smoothing factor (0 < alpha < 1)

def calcDistance(pointA, pointB):
    # Calculate the Euclidean distance between point A and B
    dist = hypot((pointA[0]-pointB[0]), (pointA[1]-pointB[1]))
    return dist

def mouthTrack(faces, frame):
    global previous_dist  # Use global to retain the previous distance value across frames

    for face in faces:
        facialLandmarks = predictor(frame, face)

        # Outer lip top point
        outerTopX = facialLandmarks.part(51).x
        outerTopY = facialLandmarks.part(51).y

        # Outer lip bottom point
        outerBottomX = facialLandmarks.part(57).x
        outerBottomY = facialLandmarks.part(57).y

        # Calculate the distance between the top and bottom lip
        current_dist = calcDistance((outerTopX, outerTopY), (outerBottomX, outerBottomY))

        # Apply smoothing using weighted average
        if previous_dist is not None:
            current_dist = alpha * current_dist + (1 - alpha) * previous_dist

        previous_dist = current_dist  # Update the previous distance with the current value

        # Determine if the mouth is open or closed
        if current_dist > 23:
            cv2.putText(frame, "Mouth Open", (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return "Mouth Open"
        else:
            cv2.putText(frame, "Mouth Closed", (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return "Mouth Closed"
    
    return -1
