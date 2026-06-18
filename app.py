import os
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predict import predict_details

app = Flask(__name__)

# Configure upload folder and allowed extensions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, 'best_model_full.pth'),
    os.path.join(BASE_DIR, 'best_model_finetuned.pth'),
    os.path.join(BASE_DIR, 'best_model.pth'),
]
MODEL_PATH = os.environ.get('FAKE_IMAGE_MODEL_PATH')
if not MODEL_PATH:
    MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), MODEL_CANDIDATES[-1])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_path': os.path.basename(MODEL_PATH),
        'model_available': os.path.exists(MODEL_PATH),
        'allowed_extensions': sorted(ALLOWED_EXTENSIONS)
    })


@app.route('/predict', methods=['POST'])
def run_prediction():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    if file and allowed_file(file.filename):
        # Generate a unique secure filename to prevent collisions
        ext = file.filename.rsplit('.', 1)[1].lower()
        original_name = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_name}" if original_name else f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # Save temporary file
            file.save(filepath)
            
            # Run deep learning inference
            details = predict_details(filepath, model_path=MODEL_PATH)
            
            # Clean up the file
            try:
                os.remove(filepath)
            except OSError:
                pass
            
            return jsonify({
                'success': True,
                'classification': details['prediction'],
                'confidence': float(details['confidence']),
                'real_probability': float(details['real_probability']),
                'fake_probability': float(details['fake_probability']),
                'model': os.path.basename(MODEL_PATH)
            })
            
        except Exception as e:
            # Clean up if file exists
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Unsupported file type. Use JPG, PNG, WEBP, or BMP.'}), 400

if __name__ == '__main__':
    # Run the server on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
