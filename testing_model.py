import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model = tf.keras.models.load_model('model/dog_breed_classifier.h5')

def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# specify path to image
image_path = 'TestData/images.jpeg'

#preprocess the image
image_array = preprocess_image(image_path)

# Get the predicted probabilities for each class
predictions = model.predict(image_array)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)[0]

train_dir = os.path.join('dataset', 'train')
class_names = sorted(os.listdir(train_dir))
breed_mapping = {i: (breed[10:].replace("_", " ") if breed[10].isupper() else breed[10:].replace("_", " ").capitalize()) for i, breed in enumerate(class_names)}


print(f"Predicted class: {breed_mapping[predicted_class]}")

for breedName in sorted(breed_mapping.values()):
    print(breedName)


