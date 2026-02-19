from flask import Flask, send_from_directory, request, jsonify, send_file
from flask_cors import CORS
import os
from main import generate_handwriting

app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trainedmodel.pt")
DATA_PATH = os.path.join(BASE_DIR, "data/")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BIAS = 10.0

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        text = data.get('text', '')
        animate = data.get('animate', False)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Generate handwriting and get the output path
        output_path = generate_handwriting(
            input_text=text,
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            animate=animate,
            bias=BIAS
        )

        # Extract the filename from the output path
        filename = os.path.basename(output_path)

        # Serve the generated file
        return send_file(output_path, mimetype='image/' + ('gif' if animate else 'png'))

    except Exception as e:
        print(f"Error in /generate: {str(e)}")  # Log the error to the terminal
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)