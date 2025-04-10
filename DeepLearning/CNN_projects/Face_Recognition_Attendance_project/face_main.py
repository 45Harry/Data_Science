import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to training images
path = 'Training_images'

images = []
classNames = []
myList = os.listdir(path)
print("Files found in directory:", myList)

# Load images with error checking
for cl in myList:
    img_path = os.path.join(path, cl)
    curImg = cv2.imread(img_path)
    if curImg is None:
        print(f"Warning: Could not load image {img_path}")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

if not images:
    raise ValueError("No valid images found in the directory. Program exiting.")

print("Successfully loaded images for:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb)
            if len(encodings) > 0:
                encodeList.append(encodings[0])
            else:
                print("Warning: No face found in one of the training images")
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(f"Attendance marked for {name} at {dtString}")

# Generate encodings
encodeListKnown = findEncodings(images)
if not encodeListKnown:
    raise ValueError("No valid face encodings were generated. Check your training images.")
print('Encoding Complete. Found', len(encodeListKnown), 'valid face encodings.')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break
    
    # Resize for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            # Scale back up face locations since the frame was scaled down
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw rectangle around face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()