import cv2
import numpy as np
import time
from collections import deque

# Load YOLO model
net = cv2.dnn.readNet("object_detection_model/weights/yolov3-tiny.weights", "object_detection_model/config/yolov3-tiny.cfg")

# Load class labels
label_classes = []
with open("object_detection_model/objectLabels/coco.names", "r") as file:
    label_classes = [name.strip() for name in file.readlines()]

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[layer - 1] for layer in net.getUnconnectedOutLayers()]

# Initialize colors and font
colors = np.random.uniform(0, 255, size=(len(label_classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

# Initialize deque for temporal smoothing
smooth_boxes = deque(maxlen=10)  # Store last 10 frame detections

def detectObject(frame):
    labels_this_frame = []

    height, width, _ = frame.shape

    # Resize frame to 416x416 (YOLO expects this size for input)
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Measure time for net.forward()
    start_time = time.time()
    try:
        outs = net.forward(output_layers)
    except Exception as e:
        print(f"Error during net.forward: {e}")
        return []

    end_time = time.time()
    print(f"net.forward() took {end_time - start_time} seconds")

    class_ids = []
    confidences = []
    boxes = []

    # Process each output layer
    for out in outs:
        for detection in out:
            if detection.shape[0] < 85:
                print(f"Skipping detection with unexpected shape: {detection.shape}")
                continue

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        indexes = indexes.flatten()
        
        frame_boxes = []
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(label_classes[class_ids[i]])
            color = colors[class_ids[i]]

            frame_boxes.append([x, y, w, h, class_ids[i], confidences[i]])

            labels_this_frame.append((label, confidences[i]))

        # Store current frame boxes for smoothing
        smooth_boxes.append(frame_boxes)

        # Calculate average boxes over last few frames
        avg_boxes = np.mean([np.array(box) for frame_box in smooth_boxes for box in frame_box], axis=0)

        for avg_box in avg_boxes.reshape(-1, 6):
            x, y, w, h, class_id, confidence = map(int, avg_box)
            color = colors[class_id]
            label = str(label_classes[class_id])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    return labels_this_frame
