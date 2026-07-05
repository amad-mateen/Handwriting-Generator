from flask import Flask, send_from_directory, request, jsonify, send_file
from flask_cors import CORS
import os
import logging

from src.config import MODEL_PATH, DATA_DIR, OUTPUT_DIR, DEFAULT_BIAS
from src.inference import generate_handwriting, get_or_load_resources

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend', static_url_path='/')
CORS(app)

# Pre-load/warm-up model weights and vocabulary at server startup
try:
    logger.info("Warming up Handwriting Synthesis model and vocabulary in memory...")
    get_or_load_resources(MODEL_PATH, DATA_DIR)
    logger.info("Warming up complete. System ready.")
except Exception as e:
    logger.error(f"Critical error: Failed to warm-up model weights at startup: {e}")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        animate = data.get('animate', False)
        style_preset = data.get('style_preset', None)  # Extract style preset (integer index or None)
        
        # Parse bias argument from request, falling back to default configuration bias
        try:
            bias = float(data.get('bias', DEFAULT_BIAS))
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid bias parameter value. Bias must be a float.'}), 400

        if not text:
            return jsonify({'error': 'No text parameter provided in request.'}), 400

        logger.info(f"Received generation request: text='{text}', animate={animate}, bias={bias}, style_preset={style_preset}")

        # Perform handwriting synthesis and get output path
        output_path = generate_handwriting(
            input_text=text,
            model_path=MODEL_PATH,
            data_path=DATA_DIR,
            output_dir=OUTPUT_DIR,
            animate=animate,
            bias=bias,
            style_preset=style_preset
        )

        mimetype = 'image/' + ('gif' if animate else 'png')
        return send_file(output_path, mimetype=mimetype)

    except ValueError as val_err:
        # Catch validation failures (e.g. out-of-vocab characters) and return a 400 Bad Request
        logger.warning(f"Request validation failed: {val_err}")
        return jsonify({'error': str(val_err)}), 400
        
    except Exception as e:
        logger.error(f"Unhandled error in generation flow: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error processing handwriting generation.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)