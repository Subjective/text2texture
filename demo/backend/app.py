"""
Flask Backend for Text2Texture.

Endpoints:
- /api/generate_image: Generates an image from a text prompt using OpenAI DALL-E.
- /api/predict_mask: Predicts a segmentation mask using SAM2 based on image and points.
- /api/generate_masks: Generates masks from an image using Florence 2 + HQ-SAM.
- /api/generate_model: Generates a 3D model (STL/PLY) from a heightmap derived from an input image,
                 optionally considering provided masks.
- /outputs/<filename>: Serves generated 3D model files.
"""
import logging
import os
from flask import Flask, send_from_directory
from flask_cors import CORS

# Import configuration, models, and blueprints
import config
import models
from routes.generation import generation_bp
from routes.masks import masks_bp
from routes.static import static_bp # Import the new static blueprint

# --- Basic Logging Setup ---
# Configure logging level and format
# More advanced configuration (e.g., file logging) could be added here
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s %(levelname)s: %(module)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Create Flask App ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    CORS(app) # Enable Cross-Origin Resource Sharing

    # --- Ensure Folders Exist ---
    try:
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
        logger.info(f"Ensured directories exist: {config.UPLOAD_FOLDER}, {config.OUTPUT_FOLDER}")
    except OSError as e:
        logger.error(f"Error creating directories: {e}", exc_info=True)
        # Depending on severity, might want to exit or handle differently
        exit(1) # Exit if we can't create essential folders

    # --- Register Blueprints ---
    app.register_blueprint(generation_bp)
    app.register_blueprint(masks_bp)
    app.register_blueprint(static_bp)
    logger.info("Registered API and static blueprints.")

    return app

# --- Main Execution ---
if __name__ == "__main__":
    app = create_app()

    # Check if we are in the reloader's main process
    # Only load models in the actual worker process to avoid double loading
    is_reloader_main_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if not config.FLASK_DEBUG or is_reloader_main_process:
        logger.info("Loading models...")
        # Use the centralized model loading function
        models.load_all_models()
        logger.info("Model loading complete (or attempted).")
    else:
        logger.info("Skipping model loading in Flask reloader parent process.")

    # Start Flask development server using config values
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT} (Debug: {config.FLASK_DEBUG})...")
    # Let app.run handle the reloader based on debug_mode
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
    logger.info("Flask backend stopped.")
