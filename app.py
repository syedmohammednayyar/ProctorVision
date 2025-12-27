import cv2
import time
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection 
import winsound
from datetime import datetime

# For Beeping
frequency = 2500
duration = 1000

# OpenCV videocapture for the webcam
cam = cv2.VideoCapture(0)

# Check if camera is opened
if not cam.isOpened():
    print("Error: Camera not found.")
    exit()

# Face Count If-else conditions
def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(5)
        remark = "Multiple faces have been detected."
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        time.sleep(3)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark

# Main function 
def proctoringAlgo():
    blinkCount = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Reading the Current time
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        print("Current Time is:", current_time)

        # Returns the face count and will detect the face.
        faceCount, faces = detectFace(frame)
        remark = faceCount_detection(faceCount)
        print(remark)

        if faceCount == 1:
            # Blink Detection
            blinkStatus = isBlinking(faces, frame)
            print(blinkStatus[2])

            if blinkStatus[2] == "Blink":
                blinkCount += 1
                print(f"Blink count: {blinkCount}")

            # Gaze Detection
            eyeStatus = gazeDetection(faces, frame)
            print(eyeStatus)

            # Mouth Position Detection
            mouthStatus = mouthTrack(faces, frame)
            print(mouthStatus)

            # Object detection using YOLO
            objectName = detectObject(frame)
            print(objectName)

            if len(objectName) > 1:
                time.sleep(4)
                winsound.Beep(frequency, duration)
                continue

            # Head Pose estimation
            headPoseStatus = head_pose_detection(faces, frame)
            print(headPoseStatus)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    proctoringAlgo()

