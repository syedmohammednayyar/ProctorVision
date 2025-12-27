# ProctorVision

An intelligent AI-powered online exam monitoring system that leverages advanced computer vision techniques to detect suspicious activities and ensure fair examinations in remote environments.

## 📋 Project Overview

**ProctorVision** is a comprehensive AI-powered online exam monitoring system designed to detect suspicious activities and ensure exam integrity in remote learning environments. This project utilizes advanced computer vision techniques to monitor examinee behavior, detect prohibited activities, and maintain fair examination standards. By combining facial detection, eye tracking, head pose estimation, and behavioral analysis algorithms, it provides real-time monitoring capabilities with minimal latency.

The system processes live video feeds from webcams, detects facial landmarks, monitors eye movement patterns, identifies suspicious behaviors such as unauthorized eye movement, head turning, or object detection outside the exam area, and logs all activities with timestamps. This makes it ideal for educational institutions, online certification programs, and remote assessment platforms that require robust proctoring solutions.

---

## ✨ Features

- **Real-Time Eye Blink Detection**: Detects eye blinks as they occur with minimal latency
- **Facial Detection**: Robust face detection using deep learning models
- **Eye Tracking**: Precise tracking of eye position and movement
- **Head Pose Estimation**: Determines head orientation for improved eye tracking accuracy
- **Activity Monitoring**: Tracks user activity status and fatigue indicators
- **Object Detection**: Integrates YOLO-based object detection capabilities
- **Multi-Face Support**: Handles multiple faces in a single frame
- **Mouth Tracking**: Companion mouth detection for enhanced facial analysis
- **Video Input Support**: Process real-time webcam feeds or pre-recorded videos
- **Performance Optimized**: Lightweight models (YOLOv3-Tiny) for real-time processing

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **OpenCV** | Real-time computer vision processing |
| **dlib** | Facial landmark detection and shape prediction |
| **NumPy** | Numerical computations and array operations |
| **YOLOv3** | Object and face detection |
| **Camera/Webcam** | Real-time video input |

### Key Models
- **Facial Landmark Detection**: dlib's 68-point facial landmark predictor
- **Object Detection**: YOLOv3 and YOLOv3-Tiny models (COCO dataset)
- **Real-time Processing**: Optimized for CPU and GPU execution

---

## 📥 Installation Instructions

### Prerequisites
- **Python 3.7+** installed on your system
- **pip** package manager
- **Webcam** or video file for input
- Minimum 4GB RAM recommended

### Step 1: Clone or Download the Repository
```bash
git clone https://github.com/yourusername/RealTime-Eye-Blink-Detection.git
cd RealTime-Eye-Blink-Detection
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models
The project includes pre-trained models for:
- YOLOv3 weights (located in `object_detection_model/weights/`)
- dlib shape predictor model (required for facial landmark detection)

Ensure all model files are in their respective directories before running the application.

---

## 🚀 How to Run the Project

### Running the Main Application
```bash
python app.py
```

The application will:
1. Initialize the webcam
2. Detect faces in the video feed
3. Identify eye regions and track eye movement
4. Detect blink events in real-time
5. Display results on the video stream

### Running Individual Modules
Each module can be executed independently for testing:

```bash
# Test blink detection
python blink_detection.py

# Test eye tracking
python eye_tracker.py

# Test facial detection
python facial_detections.py

# Test head pose estimation
python head_pose_estimation.py

# Test mouth tracking
python mouth_tracking.py

# Test object detection
python object_detection.py
```

### Keyboard Controls
- **'q'**: Quit the application
- **'s'**: Save current frame
- **'r'**: Reset counters
- **'p'**: Pause/Resume video

---

## 📁 Folder Structure

```
RealTime-Eye-Blink-Detection/
│
├── app.py                          # Main application entry point
├── blink_detection.py              # Core blink detection algorithm
├── eye_tracker.py                  # Eye tracking module
├── facial_detections.py            # Facial detection module
├── head_pose_estimation.py         # Head pose estimation algorithm
├── mouth_tracking.py               # Mouth detection and tracking
├── object_detection.py             # YOLO-based object detection
│
├── object_detection_model/         # Object detection models
│   ├── config/
│   │   ├── yolov3.cfg             # YOLOv3 configuration
│   │   └── yolov3-tiny.cfg        # YOLOv3-Tiny configuration
│   ├── objectLabels/
│   │   └── coco.names             # COCO dataset class names
│   └── weights/
│       ├── yolov3.weights         # Full YOLOv3 weights
│       └── yolov3-tiny.weights    # YOLOv3-Tiny weights
│
├── shape_predictor_model/          # dlib facial landmark predictor
│
├── activity.txt                    # Activity logs and statistics
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── __pycache__/                    # Cached Python files
```

---

## 💡 Usage Instructions

### Basic Usage

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Position Your Face**
   - Ensure your face is clearly visible to the webcam
   - Maintain adequate lighting for better detection accuracy

3. **Monitor Output**
   - Real-time blink count will be displayed on the video feed
   - Green boxes indicate detected faces and eyes
   - Blink events are logged with timestamps

### Advanced Usage

**Process a Video File**
```python
from blink_detection import BlinkDetector

detector = BlinkDetector()
detector.process_video('path/to/video.mp4')
```

**Integrate into Your Application**
```python
import cv2
from eye_tracker import EyeTracker
from blink_detection import BlinkDetector

cap = cv2.VideoCapture(0)
eye_tracker = EyeTracker()
blink_detector = BlinkDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect eyes and blinks
    eyes = eye_tracker.detect(frame)
    blinks = blink_detector.detect(eyes)
    
    cv2.imshow('Blink Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Configuration

You can adjust detection parameters in the respective module files:
- **Blink Threshold**: Sensitivity for blink detection
- **Face Detection Confidence**: Minimum confidence for face detection
- **Frame Skip Rate**: Process every nth frame for performance optimization

---

## 📊 Example Output

When running the application, you will see:

- **Live Video Feed**: A window displaying the webcam stream
- **Face Bounding Box**: Green rectangles around detected faces
- **Eye Regions**: Highlighted eye detection areas
- **Blink Counter**: Real-time count of detected blinks
- **FPS Indicator**: Current frames per second
- **Head Pose**: Estimated head orientation (yaw, pitch, roll angles)

**Sample Console Output:**
```
Face detected at: (120, 80) - (320, 380)
Left Eye detected at: (150, 140)
Right Eye detected: (280, 140)
Blink detected! Total blinks: 1
Blink detected! Total blinks: 2
FPS: 28.5
```

---

## 🚧 Future Enhancements

- **Eye Gaze Tracking**: Precise gaze direction estimation
- **Drowsiness Alert System**: Alert system based on blink frequency and head position
- **Multi-threaded Processing**: Improved performance with concurrent processing
- **Deep Learning Models**: Integration of advanced CNN-based blink classifiers
- **Pupil Diameter Tracking**: Measure pupil size changes over time
- **Emotion Detection**: Combine with facial expression analysis
- **Web Interface**: Flask/Django-based web application for remote monitoring
- **Mobile Deployment**: Optimize for mobile devices and edge computing
- **Performance Metrics Dashboard**: Detailed analytics and visualization
- **GPU Optimization**: CUDA/cuDNN support for faster processing
- **Data Logging**: Comprehensive blink event logging and analysis
- **Calibration Tool**: User-specific calibration for improved accuracy

---

## 📝 License

You are free to use, modify, and distribute this software for educational and commercial purposes, with appropriate attribution.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**: Create your own fork of this project
2. **Create a Feature Branch**: `git checkout -b feature/YourFeature`
3. **Commit Your Changes**: `git commit -m 'Add YourFeature'`
4. **Push to the Branch**: `git push origin feature/YourFeature`
5. **Submit a Pull Request**: Describe your changes clearly

### Code Style
- Follow PEP 8 guidelines for Python code
- Add docstrings to functions and classes
- Include comments for complex logic

---

## 📧 Support & Contact

- **Issues**: Report bugs and issues on the GitHub Issues page
- **Discussions**: Start discussions for feature requests and ideas
- **Email**: syedmohammednayyar@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/syedmohammednayyar/

---

## 🙏 Acknowledgments

- **dlib library** for facial landmark detection
- **OpenCV community** for computer vision tools
- **YOLO** authors for object detection framework
- All contributors and testers

---

## 📚 References & Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [dlib Face Recognition](http://dlib.net/python/index.html)
- [YOLO Object Detection](https://pjreddie.com/darknet/yolo/)
- [Computer Vision Techniques](https://en.wikipedia.org/wiki/Computer_vision)

---

**Last Updated**: December 27, 2025  
**Version**: 1.0.0

---

*For the best experience, ensure adequate lighting and maintain a distance of 30-60 cm from the webcam.*
