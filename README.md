# Military Vehicle Detection

This project focuses on detecting various types of military vehicles using a Convolutional Neural Network (CNN). The model is trained on a dataset of labeled images and classifies vehicles into distinct categories such as tanks, armored personnel carriers, and anti-aircraft vehicles.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to accurately classify military vehicles using deep learning. The project includes training a CNN on a dataset of military vehicle images, testing the model's accuracy, and using the trained model to make predictions on new images.

### Categories

The model classifies images into the following categories:

1. Anti-aircraft
2. Armored combat support vehicles
3. Armored personnel carriers
4. Infantry fighting vehicles
5. Light armored vehicles
6. Mine-protected vehicles
7. Prime movers and trucks
8. Self-propelled artillery
9. Light utility vehicles
10. Tanks

## Installation

To run this project, ensure you have Python installed along with the following dependencies:

```bash
pip install tensorflow opencv-python
```

You can also clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/military-vehicle-detection.git
cd military-vehicle-detection
pip install -r requirements.txt
```

## Dataset

The dataset consists of images of military vehicles stored in different folders, each representing a specific class. The dataset can be found in the `/train` directory of the Google Drive. It is split into training and validation sets.

You can load the dataset using TensorFlow's `ImageDataGenerator`, as shown in the code:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '/path/to/your/dataset'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

## Model Architecture

The model uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Here's a summary of the architecture:

- **Conv2D layers**: Three convolutional layers with 32, 64, and 128 filters, respectively, each followed by max pooling.
- **Dense layers**: Two dense layers, including one with 512 units and ReLU activation, and a final softmax layer for classification.
- **Dropout layer**: A dropout layer with a 50% rate to reduce overfitting.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Training

The model is trained using the Adam optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

The number of epochs can be adjusted as needed. The training and validation accuracy/loss are recorded during training.

## Results

Once training is complete, the model can be used to predict the class of military vehicles in new images. Example output includes:

- **Predicted class**: Tanks
- **Accuracy**: 90% (This is an example; actual accuracy may vary depending on training results)

## Usage

After training, the model can be used to classify new images. Here’s how you can use the model to make predictions:

```python
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Load and preprocess image
img_path = '/path/to/new/image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to vehicle type
classes = {0: 'Anti-aircraft', 1: 'Armored combat support vehicles', ...}  # List all classes
predicted_class_name = classes[predicted_class]
print(f'Predicted class: {predicted_class_name}')
```

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request or open an issue on GitHub. Contributions for improving the model, adding new features, or providing better datasets are welcome.


