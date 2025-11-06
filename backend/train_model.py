import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Set paths
tumor_path = 'data/tumors'
non_tumor_path = 'data/non tumors'

# Temp folders for binary classification format
base_path = 'data'
train_path = os.path.join(base_path, 'TRAIN')
val_path = os.path.join(base_path, 'VAL')
test_path = os.path.join(base_path, 'TEST')

# Create directories
for folder in [train_path, val_path, test_path]:
    for class_name in ['tumor', 'non_tumor']:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

# Collect image paths
def collect_images_from_dirs(parent_path):
    image_paths = []
    for folder in os.listdir(parent_path):
        full_folder = os.path.join(parent_path, folder)
        if os.path.isdir(full_folder):
            for img in os.listdir(full_folder):
                image_paths.append(os.path.join(full_folder, img))
    return image_paths

tumor_imgs = collect_images_from_dirs(tumor_path)
non_tumor_imgs = collect_images_from_dirs(non_tumor_path)

# Split data
def split_and_copy(img_paths, label):
    train, test = train_test_split(img_paths, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    def copy_to_set(image_list, target_dir):
        for img_path in image_list:
            if os.path.exists(img_path):  # safety
                shutil.copy(img_path, os.path.join(target_dir, label, os.path.basename(img_path)))
    
    copy_to_set(train, train_path)
    copy_to_set(val, val_path)
    copy_to_set(test, test_path)

split_and_copy(tumor_imgs, 'tumor')
split_and_copy(non_tumor_imgs, 'non_tumor')

# Generators
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, rotation_range=20)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(128, 128), class_mode='binary')
val_data = val_gen.flow_from_directory(val_path, target_size=(128, 128), class_mode='binary')

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)
model.save("backend/model/cnn_model.h5")

