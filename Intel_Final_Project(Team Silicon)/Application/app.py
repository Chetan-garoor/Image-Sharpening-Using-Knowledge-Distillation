# app.py (Located in the root of your project, e.g., your_image_sharpening_project/app.py)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import os
import webbrowser
import threading
from pytorch_msssim import ssim # NEW: Import ssim

# --- 1. Flask App Setup ---
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- 2. Model Definition (MUST match your trained StudentModel exactly) ---
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.down2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )

        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bottleneck(d2)
        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)
        return self.final(d2)

# --- 3. Load Trained Model ---
MODEL_PATH = 'best_student (5).pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend will attempt to use device: {device}")

model = None
try:
    model = StudentModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Student model loaded successfully from '{MODEL_PATH}' on {device}!")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_PATH}' not found. Please ensure it is placed in the '{os.path.basename(os.getcwd())}' directory alongside 'app.py'.")
    print("The API will return an error for image sharpening requests until the model is loaded.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading the model: {e}")
    print("Please check your model file and definition. API requests will fail.")

# --- 4. Image Preprocessing and Postprocessing Transforms ---
preprocess_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- 5. API Endpoint for Image Sharpening ---
@app.route('/sharpen', methods=['POST'])
def sharpen_image():
    if model is None:
        return jsonify({'error': 'Server model not initialized. Please check backend startup logs.'}), 500
    if not request.json or 'image_data' not in request.json:
        return jsonify({'error': 'Invalid request: Missing "image_data" in JSON body.'}), 400

    image_data_b64 = request.json['image_data']

    try:
        image_bytes = base64.b64decode(image_data_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Store original input tensor for SSIM calculation later
        original_input_tensor = preprocess_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(original_input_tensor) # Use original_input_tensor for model prediction

        # Post-process output tensor
        output_tensor_cpu = output_tensor.squeeze(0).cpu() # Remove batch dim, move to CPU
        
        # Denormalize tensors to [0, 1] for SSIM calculation and image saving
        original_input_denorm = (original_input_tensor.squeeze(0).cpu() + 1) / 2
        output_denorm = (output_tensor_cpu + 1) / 2

        # Calculate SSIM between the original blurry input and the sharpened output
        # A lower SSIM here indicates less similarity, meaning more transformation/sharpening
        ssim_value = ssim(
            original_input_denorm.unsqueeze(0), # Add batch dim for ssim function
            output_denorm.unsqueeze(0), # Add batch dim for ssim function
            data_range=1.0, size_average=True
        ).item()

        confidence_message = "Sharpening Processed Successfully!"

        # Convert sharpened output to PIL Image for encoding
        output_img = transforms.ToPILImage()(output_denorm.clamp(0, 1))

        buffered = io.BytesIO()
        output_img.save(buffered, format="JPEG", quality=90)
        sharpened_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'sharpened_image_data': sharpened_image_b64,
            'ssim_score': f"{ssim_value:.4f}", # Format SSIM to 4 decimal places
            'confidence': confidence_message
        })

    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({'error': f'An internal server error occurred during image processing: {str(e)}'}), 500

# --- 6. Route to serve the HTML frontend ---
@app.route('/')
def index():
    return render_template('index.html')

# --- 7. Run the Flask App and open browser (FIXED for single tab) ---
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    url = f"http://{host}:{port}"

    # This check ensures the browser opens only once, not on reloader restarts
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        def open_browser():
            webbrowser.open_new(url)
        threading.Timer(1.5, open_browser).start() # 1.5 second delay

    app.run(host=host, port=port, debug=True)
