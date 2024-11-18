# HITL Animal Classification System

This project is a Human-In-The-Loop (HITL) animal classification system. The goal is to iteratively improve a machine learning model's classification accuracy through user interaction. Users select an animal category, and the model attempts to identify the correct image. User feedback is then used to improve the model over time.

## Project Structure

The project contains both a frontend (Vue.js) and a backend (Python/Flask) that interact to provide an iterative learning experience.

### Frontend

- **Technology**: Vue.js
- **Purpose**: Displays animal images for classification and allows users to interact with the model by selecting a category.
- **Key Features**:
  - Users can save model weights, reset the model, or use a pre-trained model.
  - The model's guess and its certainty are displayed after each prediction.

### Backend

- **Technology**: Python (Flask, PyTorch)
- **Purpose**: Handles model training, prediction, and state management.
- **Endpoints**:
  - `/predict` - Makes predictions based on selected images.
  - `/train` - Updates the model using user feedback.
  - `/reset_model` - Resets the model to initial weights.
  - `/save_model_weights` - Saves the current model weights.
  - `/use_trained_model` - Loads pre-trained model weights.

## Setup Instructions

### Prerequisites

- **Python 3.x**
- **Node.js and npm**
- **Git**
- **Virtual Environment** (for Python)

### Backend Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>/backend
   ```
2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Flask Backend**:
   ```bash
   python app.py
   ```

### Frontend Setup

1. **Navigate to Frontend Directory**:
   ```bash
   cd <repository_folder>/frontend
   ```
2. **Install Dependencies**:
   ```bash
   npm install
   ```
3. **Run Development Server**:
   ```bash
   npm run serve
   ```

### Deploying to Server

To deploy the project on a server:

1. **Upload the Repository to Your Server**.
2. **Install Backend Requirements** as described above on the server.
3. **Build the Frontend**:
   ```bash
   npm run build
   ```
4. **Copy the \*\***`dist`\***\* Folder** to your server’s web directory (e.g., `/public_html/hitl`).
5. **Run Backend in Production** using a production server such as `gunicorn`.
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

## Usage

- **Interact with the System**: Use the frontend to select an animal category and provide feedback to improve the model.
- **Save or Reset the Model**: Use control buttons to save weights, reset the model, or load pre-trained weights.

## Important Files

- **Frontend** (`frontend/`): Vue.js components, `package.json` for dependencies.
- **Backend** (`backend/`): Flask API (`app.py`), `requirements.txt` for Python dependencies, model weights in `weights/` folder.
- **README.md**: This documentation.

## Running with Git

To ensure all relevant files are tracked and available:

1. **Add \*\***`.gitignore`\*\* to ignore unnecessary files like `node_modules/` and Python virtual environments (`venv/`).
2. **Push Changes** to your remote repository to ensure everyone has access to updated code and setup instructions.

## `.gitignore` Example

```
# Node.js
node_modules/
dist/

# Python
__pycache__/
venv/
*.pyc

# Model Weights and Sensitive Files
weights/
*.pth
.env
```

## License

This project is open-source and available under the MIT License.

## Contact

For any questions or contributions, please contact Martin at marg@kea.dk
