from google.colab import drive
drive.mount('/content/drive')




!git clone https://github.com/TheDunkinArt/face_detect_python.git
%cd face_detect_python






import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from google.colab.patches import cv2_imshow

# Load the face detection and mask detection models
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model = load_model("mask_recog.h5")

# Function to detect faces and determine if they are wearing masks
def face_mask_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    preds = []
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)
        
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
        
        for pred in preds:
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    return frame

# Directory to start scanning for images
root_dir = 'Dataset'

# Loop through each subdirectory and file
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Only process image files (e.g., png, jpg)
        if file.endswith('.png') or file.endswith('.jpg'):
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            
            # Read the image
            input_image = cv2.imread(file_path)
            
            # Apply the face mask detector
            output = face_mask_detector(input_image)
            
            # Display the result using cv2_imshow (Google Colab specific)
            cv2_imshow(output)

            # Optionally, save the output image to a directory
            # output_file_path = os.path.join(subdir, "output_" + file)
            # cv2.imwrite(output_file_path, output)
            
            # Add a delay to view the image (optional, adjust as needed)
            cv2.waitKey(1000)  # Wait for 1 second before moving to the next image

# Clean up windows if using local environment (not needed in Colab)
# cv2.destroyAllWindows()
