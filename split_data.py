import os
import random
import shutil

# Set the configuration for train and validation split
source_dir = 'images'
output_dir = 'dataset'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'validation')
split_ratio = 0.8

# Create the output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through each breed folder
for breed in os.listdir(source_dir):
    breed_folder = os.path.join(source_dir, breed)
    if os.path.isdir(breed_folder):
        train_bread_dir = os.path.join(train_dir, breed)
        val_bread_dir = os.path.join(val_dir, breed)
        os.makedirs(train_bread_dir, exist_ok=True)
        os.makedirs(val_bread_dir, exist_ok=True)

        # List all the images in the breed folder
        images = [img for img in os.listdir(breed_folder) 
                  if os.path.isfile(os.path.join(breed_folder, img))]
        
        # Shuffle the images
        random.shuffle(images)

        # Determine the split index
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Copy the images to the train and validation folders

        for img in train_images:
            src = os.path.join(breed_folder, img)
            dst = os.path.join(train_bread_dir, img)
            shutil.copy(src, dst)

        for img in val_images:
            src = os.path.join(breed_folder, img)
            dst = os.path.join(val_bread_dir, img)
            shutil.copy(src, dst)

        print(f"Processed '{breed}': {len(train_images)} training images and {len(val_images)} validation images")

        