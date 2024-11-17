from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

app = Flask(__name__)

# Define a mapping for category names to indices
categories_mapping = {
    'Mammal': 0,
    'Bird': 1,
    'Reptile': 2,
    'Fish': 3,
    'Amphibian': 4,
    'Insect': 5,
    'Invertebrate': 6
}
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Define a simple neural network model using PyTorch
class SimpleAnimalClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleAnimalClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
input_size = 16  # Adjust input size based on your actual feature length from the previous model
num_classes = 7  # Number of categories (e.g., Mammal, Bird, etc.)
model = SimpleAnimalClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load model weights if they exist
from datetime import datetime

# Create directory for weights if it doesn't exist
weights_dir = 'weights'
os.makedirs(weights_dir, exist_ok=True)

model_weights_path = os.path.join(weights_dir, 'model_weights.pth')
if os.path.exists(model_weights_path):
    try:
        model.load_state_dict(torch.load(model_weights_path))
    except RuntimeError as e:
        print(f"Warning: {e}. The model architecture has changed. Loading weights is skipped.")

# Train endpoint to receive the selected images and attributes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print('Received data:', data)

    # Get the attributes of the three displayed images
    selected_images = data.get('selected_images')  # A list of dictionaries with image attributes
    selected_category = data.get('selected_category')
    category_index = categories_mapping.get(selected_category)

    if len(selected_images) != 3 or category_index is None:
        return jsonify({'error': 'Invalid input data'}), 400

    attributes_list = [image['attributes'] for image in selected_images]

    # Forward pass with each set of attributes to see which one fits best
    best_match = None
    best_loss = float('inf')
    guessed_index = None

    for idx, attributes in enumerate(attributes_list):
        input_features = torch.tensor([attributes], dtype=torch.float32)
        output = model(input_features)
        loss = criterion(output, torch.tensor([category_index], dtype=torch.long))

        # Track the best match (lowest loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_match = idx

    guessed_index = best_match
    print(f'Model guessed index: {guessed_index}')

    # Save the updated model weights
    torch.save(model.state_dict(), model_weights_path)

    response = {
        'message': 'Model has made a guess',
        'guessed_index': guessed_index
    }
    return jsonify(response)

# Save model weights and start fresh
@app.route('/save_model', methods=['POST'])
def save_model():
    # Copy current weights to a new file with a timestamp
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    weights_filename = f'model_weights_{current_datetime}.pth'
    weights_path = os.path.join(weights_dir, weights_filename)
    if os.path.exists(model_weights_path):
        torch.save(model.state_dict(), weights_path)
    # Reset the model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    return jsonify({'message': f'Model weights saved successfully as {weights_filename} and model reset.'})

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    correct_image = data.get('correct_image')
    if not correct_image:
        return jsonify({'error': 'Correct image data not provided'}), 400

    selected_category = correct_image.get('category')
    category_index = categories_mapping.get(selected_category)
    if category_index is None:
        return jsonify({'error': 'Invalid category value received.'}), 400

    # Extract attributes of the correct image
    attributes = correct_image['attributes']
    input_features = torch.tensor([attributes], dtype=torch.float32)

    # Use the selected category as the correct class for training
    target = torch.tensor([category_index], dtype=torch.long)

    # Forward pass
    output = model(input_features)

    # Compute loss and update the model
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Model updated with feedback: Loss - {loss.item()}')

    # Save the updated model weights
    torch.save(model.state_dict(), model_weights_path)

    return jsonify({'message': 'Feedback received and model updated successfully'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
