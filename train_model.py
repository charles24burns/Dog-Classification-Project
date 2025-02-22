import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Define the parameters and Assiging the dataset directory
dataset_dir = 'dataset'
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Load the dataset and count the number of dog classes
train_dir = os.path.join(dataset_dir, 'train')
num_dog_breeds = len(next(os.walk(train_dir))[1])
print(f"Number of dog breeds: {num_dog_breeds}")

# Data Augmentation for training and validation dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.3,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# create the train and validation generator
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'validation'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ================================================
# Build the model using Transfer Learning
# ================================================

# Load the MobileNetV2 premodel
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_dog_breeds, activation='softmax')(x)

# Combine the base model and the top layer
model = Model(inputs=base_model.input, outputs=predictions)

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# Train the model

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model

model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'dog_breed_classifier.h5'))

print("Model training completed and saved to 'model/dog_breed_classifier.h5'")


# -------------------------------------------------------------
# Plot the training and validation accuracy and loss
# -------------------------------------------------------------

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(15, 5))

# Plot Accuracy
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot Precision
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history.history['precision'], label='Training Precision')
plt.plot(epochs_range, history.history['val_precision'], label='Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Training and Validation Precision')
plt.legend()

# Plot Recall
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history.history['recall'], label='Training Recall')
plt.plot(epochs_range, history.history['val_recall'], label='Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Training and Validation Recall')
plt.legend()

plt.tight_layout()
plt.show()


