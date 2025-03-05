import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from groq import Groq
import base64
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Initialize Groq client
client = Groq(api_key="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")  # Replace with your actual API key

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def detect_fruits_vegetables(base64_image):
    """Detect fruits and vegetables in the image"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List only the names of fruits and vegetables in this image. Return a comma-separated list with no other text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )
        
        # Extract and clean the response
        response = chat_completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        print(f"Error detecting fruits/vegetables: {e}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload with improved error handling"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the file
            file.save(filepath)
            
            # Encode and detect
            base64_image = encode_image(filepath)
            
            if base64_image is None:
                os.remove(filepath)
                return jsonify({"error": "Failed to encode image"}), 500
            
            result = detect_fruits_vegetables(base64_image)
            
            # Remove the uploaded file
            os.remove(filepath)
            
            return jsonify({"fruits_vegetables": result})
        
        except Exception as e:
            print(f"Upload error: {e}")
            # Ensure file is removed even if an error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "File upload failed"}), 500
    
    # If file extension is not allowed
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/webcam', methods=['POST'])
def webcam_image():
    """Handle webcam image"""
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({"error": "No image data"}), 400
    
    # Remove the data URL prefix
    try:
        base64_image = image_data.split(',')[1]
        
        # Save the image temporarily
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(base64_image))
        
        # Detect fruits and vegetables
        result = detect_fruits_vegetables(base64_image)
        
        # Remove the temporary file
        os.remove(filepath)
        
        return jsonify({"fruits_vegetables": result})
    
    except Exception as e:
        print(f"Webcam image error: {e}")
        return jsonify({"error": "Failed to process webcam image"}), 500

if __name__ == '__main__':
    app.run(debug=True)