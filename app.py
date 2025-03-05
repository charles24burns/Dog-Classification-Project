import os 
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import requests
import io 

# Initialize the Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = "your_secret_key"

# Configure SQLAlchemy with SQLite database for development
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dog_app.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
database = SQLAlchemy(app)

# Create the uploads folder
uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = uploads_dir

if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
else: 
    for file in os.listdir(uploads_dir):
        os.remove(os.path.join(uploads_dir, file))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the database model and store uploads and predictions
class Upload(database.Model):
    id = database.Column(database.Integer, primary_key=True)
    filename = database.Column(database.String(120), nullable=False)
    breed = database.Column(database.String(80), nullable=False)
    date_uploaded = database.Column(database.DateTime, default=datetime.now)

# create database tables (run once)
with app.app_context():
    database.create_all()

# Load the model
model = load_model('model/dog_breed_classifier.h5')

# mapping of class index to breed name
train_dir = os.path.join('dataset', 'train')
class_names = sorted(os.listdir(train_dir))
breed_mapping = {i: (breed[10:].replace("_", " ") if breed[10].isupper() else breed[10:].replace("_", " ").capitalize()) for i, breed in enumerate(class_names)}

# Reading excel file
df = pd.read_excel("Dog Breed Intelligence ranking.xlsx", sheet_name="Sheet2")

# Creating a dictionary of breed intelligence
breed_intelligence = dict(zip(df["Breed"], df["Category"]))

# Combine the Dog intelligence level with the breed mapping by assigning
# the intelligence level to the breed name using terms like brighest dogs for 
# dogs ranked 1-10, excellent dog for dogs ranked 11-26, and so on

breed_mapping_ranking = {i: f"{breed_intelligence.get(breed, 'Unknown')}" for i, breed in breed_mapping.items()}

# Get the description of the breed
def get_breed_description(breed):
    formatted_breed = breed.replace(" ", "_")
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles={formatted_breed}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page["extract"]
    else:
        return "Description not found"


def preprocess_image(image, target_size=(224, 224)):
    # Load the image
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if "file" not in request.files:
            return "No file part in the request", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Read the image file
            image = Image.open(file.stream)
            image_array = preprocess_image(image, target_size=(224, 224))
            preds = model.predict(image_array)
            breed_index = np.argmax(preds[0])
            confidence = str(int(preds[0][breed_index] * 100)) + "%"
            predicted_breed = breed_mapping.get(breed_index, "Unknown")
            predicted_breed_ranking = breed_mapping_ranking.get(breed_index, "Unknown")
            breed_description = get_breed_description(predicted_breed)

            # Save the uploaded image
            new_upload = Upload(filename=file.filename, breed=predicted_breed)
            database.session.add(new_upload)
            database.session.commit()



            image_url = url_for('static', filename='uploads/' + file.filename)
            return render_template('result.html', 
                                   breed=predicted_breed, 
                                   confidence=confidence, image=image_url, 
                                   breed_ranking = predicted_breed_ranking,
                                   description = breed_description)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)




