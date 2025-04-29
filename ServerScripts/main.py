import io
import base64
from flask import Flask, jsonify, request
import torchvision.transforms as transforms
import torch
from torch import nn
import numpy as np
from PIL import Image

# flask --app  SI_Proyecto_Final/ServerScripts/main.py run ----- Script to run the api

app = Flask(__name__)

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = DeeperCNN()
model.load_state_dict(torch.load("/home/unai/Documents/Uni/Año4/Deusto/SI/ExportedModels/v1.pth", weights_only=True))
model.eval()

classes = {0:'real', 1:'spoof'}

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('Recieved post')
        data = request.get_json()
        file = data['image']
        img = base64_to_image(file)
        img = create_image_from_bytes(img)
        print(img)
        prediction = get_prediction(img)
        

        return jsonify(prediction), 200

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize images to 224x224
        transforms.CenterCrop(224),
        transforms.ToTensor(),          # Convert images to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image)
    outputs = model.forward(tensor)
    outputs = outputs.detach().cpu().numpy()
    predicted_class = np.round(outputs)
    prediction = {
        "class": classes[int(predicted_class[0][0])],
        "value": float(outputs[0][0])
                  }
    return prediction

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = io.BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image
