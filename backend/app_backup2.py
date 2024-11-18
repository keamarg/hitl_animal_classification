from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

app = Flask(__name__)
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
        # Delete the old weights file after loading
        os.remove(model_weights_path)
    except RuntimeError as e:
        print(f"Warning: {e}. The model architecture has changed. Loading weights is skipped.")

# Train endpoint to receive the selected images and attributes
@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    print('Received data:', data)

    # Get the attributes of the three displayed images
    selected_images = data.get('selected_images')  # A list of dictionaries with image attributes
    attributes_list = [image['attributes'] for image in selected_images]

    if len(attributes_list) != 3:
        return jsonify({'error': 'Attributes for some images could not be found'}), 400

    # Forward pass with each set of attributes to see which one fits best
    best_match = None
    best_loss = float('inf')
    guessed_index = None

    for idx, attributes in enumerate(attributes_list):
        input_features = torch.tensor([attributes], dtype=torch.float32)
        output = model(input_features)
        target = torch.argmax(output, dim=1)  # The model's predicted category
        loss = criterion(output, target)

        # Track the best match (lowest loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_match = idx
            guessed_index = target.item()

    # Train the model with the actual target (best match)
    try:
        print(f'Training with attributes: {attributes_list[best_match]}')
        input_features = torch.tensor([attributes_list[best_match]], dtype=torch.float32)
        target = torch.tensor([guessed_index], dtype=torch.long)
    except IndexError:
        return jsonify({'error': 'Index error while accessing the attributes list.'}), 400

    # Forward pass
    output = model(input_features)

    # Compute loss and update the model
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Model updated: Loss - {loss.item()}, Best Match Index - {best_match}')

    # Save the updated model weights
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    weights_filename = f'model_weights_{current_datetime}.pth'
    weights_path = os.path.join(weights_dir, weights_filename)
    torch.save(model.state_dict(), weights_path)

    # Extract weights for each category (flatten to simplify structure)
    weights = model.fc.weight.detach().numpy().tolist()
    flat_weights = [sum(weight) for weight in weights]  # Sum weights for each category to simplify
    print(f'Model weights being sent: {flat_weights}')

    response = {
        'message': 'Model updated successfully',
        'loss': loss.item(),
        'guessed_index': best_match,  # This will let the frontend know which image was the model's guess
        'iteration': data.get('iteration', 0),
        'weights': flat_weights  # Return the flattened weights for visualization
    }
    return jsonify(response)

# Save model weights and start fresh
@app.route('/save_model', methods=['POST'])
def save_model():
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    weights_filename = f'model_weights_{current_datetime}.pth'
    weights_path = os.path.join(weights_dir, weights_filename)
    torch.save(model.state_dict(), weights_path)
    # Reset the model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    return jsonify({'message': f'Model weights saved successfully as {weights_filename} and model reset.'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
