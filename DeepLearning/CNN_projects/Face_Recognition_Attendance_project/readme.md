# Race Recognition Attendance Project
This project implements a real-time face recognition-based attendance system using Python, OpenCV, and the face_recognition library. The system captures video from the webcam, detects and recognizes faces, and marks attendance by recording the individual's name and the time of recognition in a CSV file.
Features

    Real-time Face Detection and Recognition: Utilizes the webcam to detect and recognize faces in real-time.

    Automatic Attendance Logging: Recognized faces are logged into an Attendance.csv file with a timestamp.

    Error Handling: Includes checks for image loading errors and cases where no faces are detected in training images.

Prerequisites

Ensure you have the following installed:

    Python 3.x

    OpenCV (cv2)

    NumPy

    face_recognition library

You can install the required Python packages using pip:

pip install opencv-python numpy face_recognition

Directory Structure

project_root/
├── Training_images/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
├── Attendance.csv
└── face_recognition_attendance.py

    Training_images/: Directory containing images of individuals to be recognized. The filename (without extension) is used as the individual's name.

    Attendance.csv: CSV file where attendance records are stored.

    face_recognition_attendance.py: Main script to run the attendance system.

Usage

    Prepare Training Images:

        Place clear images of each individual in the Training_images directory.

        Ensure each image filename corresponds to the individual's name (e.g., John_Doe.jpg).

    Run the Script:

        Execute the face_recognition_attendance.py script:

        python face_recognition_attendance.py

        The script will:

            Load and encode faces from the training images.

            Access the webcam and begin real-time face recognition.

            Draw rectangles around recognized faces and display their names.

            Log attendance in Attendance.csv with the current time.

    Exit the Program:

        To stop the webcam feed and exit the program, press the 'q' key.

Attendance Logging

The Attendance.csv file records attendance with the following format:

Name,Time
John Doe,14:35:22
Jane Smith,14:40:10
...

Each entry logs the individual's name and the time they were recognized.
Notes

    The script includes error handling for cases where images cannot be loaded or no faces are detected.

    If no valid images or face encodings are found, the program will raise an error and exit.

    Ensure the webcam is properly connected and accessible.

Acknowledgments

This project utilizes the face_recognition library for face detection and recognition, and OpenCV for image processing and webcam access.