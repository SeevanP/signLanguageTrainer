import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the directory and label mapping
#data_dir =

# List all subfolders (classes) and sort them alphabetically
class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

# Print all the folders found for debugging
print("Class folders found:", class_folders)

# Map class names to integers
label_map = {cls: idx for idx, cls in enumerate(class_folders)}
num_classes = len(label_map)  # This should be 18

# Load and preprocess images
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")  # Debugging message for invalid images
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize
    return img

images = []
image_labels = []

# Only include jpg and png images, with case-insensitive matching
valid_extensions = ['.jpg', '.jpeg', '.png']

# Loop through each class folder and its subfolders (numbered folders)
for cls in class_folders:
    cls_folder = os.path.join(data_dir, cls)
    subfolders = sorted([f for f in os.listdir(cls_folder) if os.path.isdir(os.path.join(cls_folder, f))])
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(cls_folder, subfolder)
        img_files = sorted([f for f in os.listdir(subfolder_path)
                            if os.path.isfile(os.path.join(subfolder_path, f)) 
                            and os.path.splitext(f)[1].lower() in valid_extensions])

        # Print the number of files found in each subfolder
        print(f"Processing subfolder '{cls}/{subfolder}' with {len(img_files)} files")

        for img_file in img_files:
            img_path = os.path.join(subfolder_path, img_file)
            print(f"Loading image: {img_path}")  # Debugging line to confirm image loading
            img = load_image(img_path)
            if img is not None:  # Only add if image was successfully loaded
                images.append(img)
                image_labels.append(label_map[cls])

# Convert lists to numpy arrays
images = np.array(images)
image_labels = np.array(image_labels)
image_labels = to_categorical(image_labels, num_classes)  # One-hot encode labels

# Check if images have been loaded correctly
print(f"Number of images loaded: {len(images)}")

# If no images were loaded, stop further execution
if len(images) == 0:
    print("No images were loaded. Please check the dataset and folder paths.")
    exit()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, image_labels, test_size=0.2, random_state=42)

# Define the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation loss: {loss}")
print(f"Validation accuracy: {accuracy}")

# Plot training history (Accuracy)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training history (Loss)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
