from google.colab import drive
drive.mount('/content/drive')

______________________________________________________________________


!git clone https://github.com/TheDunkinArt/face_detect_python.git
%cd face_detect_python


_______________________________________________________________________



import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

# Task 2: Load Dataset
data = []
labels = []
root_dir = 'Dataset'

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            
            if 'With-Mask' in subdir:
                labels.append(1)
            else:
                labels.append(0)

# Convert to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split dataset into training (80%) and validation (20%)
trainX, valX, trainY, valY = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Task 3: Build Model to Detect Mask and Unmask
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Task 4: Compile the Model
learning_rate = 1e-4
opt = Adam(lr=learning_rate)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Task 5: Train and Test the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

epochs = 20
batch_size = 32

history = model.fit(
    trainX, trainY, 
    validation_data=(valX, valY),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate on validation set
val_loss, val_acc = model.evaluate(valX, valY)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

# Task 6: Face Mask Detection Function
def face_mask_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
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
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    return frame

# Testing the model on a few images from the dataset
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            file_path = os.path.join(subdir, file)
            input_image = cv2.imread(file_path)
            output = face_mask_detector(input_image)
            cv2_imshow(output)
            cv2.waitKey(1000)  # Wait for 1 second before moving to the next image

# Clean up windows if using local environment (not needed in Colab)
# cv2.destroyAllWindows()
