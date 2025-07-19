import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set path to your dataset folders
dataset_path = os.path.join("archive")
image_size = (48, 48)

# Map each class folder to a label
class_names = os.listdir(os.path.join(dataset_path, 'train'))
class_names = [name for name in class_names if not name.startswith('.')]  # skip .DS_Store
class_to_label = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}

def load_data(folder):
    data = []
    labels = []
    path = os.path.join(dataset_path, folder)

    for emotion in os.listdir(path):
        emotion_folder = os.path.join(path, emotion)

        # Skip non-directories like .DS_Store
        if not os.path.isdir(emotion_folder):
            continue

        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)
            
            # Load image in grayscale and resize
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, image_size)
            data.append(img)
            labels.append(class_to_label[emotion])

    return np.array(data), np.array(labels)

# Load training and testing data
X_train, y_train = load_data('train')
X_test, y_test = load_data('test')

# Normalize image data to [0,1] and reshape
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, image_size[0], image_size[1], 1)
X_test = X_test.reshape(-1, image_size[0], image_size[1], 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))

# Save preprocessed data as .npy files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("✅ Preprocessing complete. Data saved as .npy files.")