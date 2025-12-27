import dlib
import cv2
from imutils import face_utils
import os
import time

# Path to the shape predictor model
shapePredictorModel = "shape_predictor_model/shape_predictor_68_face_landmarks.dat"

# Check if the shape predictor model file exists
if not os.path.isfile(shapePredictorModel):
    raise FileNotFoundError(f"Shape predictor model file not found at {shapePredictorModel}")

# Initialize dlib's face detector and shape predictor
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(shapePredictorModel)

def detectFace(frame):
    """
    Detect faces and facial landmarks in a given video frame.
    
    Args:
        frame (numpy.ndarray): The video frame from the camera.
        
    Returns:
        tuple: A tuple containing the count of faces and the list of detected face rectangles.
    """
    try:
        # Resize the frame to speed up detection (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Measure the time taken for face detection
        start_time = time.time()

        # Detect faces in the grayscale image
        faces = faceDetector(gray, 0)

        end_time = time.time()
        print(f"Face detection took {end_time - start_time:.2f} seconds")

        # Count the number of faces detected
        faceCount = len(faces)
        print(f"Number of faces detected: {faceCount}")

        # Loop over the detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Draw the corners of the rectangle around the face
            cv2.line(frame_resized, (x, y), (x + 20, y), (0, 255, 255), 2)
            cv2.line(frame_resized, (x, y), (x, y + 20), (0, 255, 255), 2)
            cv2.line(frame_resized, (x + w, y), (x + w - 20, y), (0, 255, 255), 2)
            cv2.line(frame_resized, (x + w, y), (x + w, y + 20), (0, 255, 255), 2)
            cv2.line(frame_resized, (x, y + h), (x + 20, y + h), (0, 255, 255), 2)
            cv2.line(frame_resized, (x, y + h), (x, y + h - 20), (0, 255, 255), 2)
            cv2.line(frame_resized, (x + w, y + h), (x + w - 20, y + h), (0, 255, 255), 2)
            cv2.line(frame_resized, (x + w, y + h), (x + w, y + h - 20), (0, 255, 255), 2)

            try:
                # Determine the facial landmarks for the face region
                facialLandmarks = shapePredictor(gray, face)
                # Convert the facial landmarks to a numpy array
                facialLandmarks = face_utils.shape_to_np(facialLandmarks)
                
                # Draw circles around the facial landmarks
                for (a, b) in facialLandmarks:
                    cv2.circle(frame_resized, (a, b), 2, (255, 255, 0), -1)
            
            except Exception as e:
                print(f"Error processing facial landmarks: {e}")

        return faceCount, faces

    except Exception as e:
        print(f"Error during face detection: {e}")
        return 0, []



