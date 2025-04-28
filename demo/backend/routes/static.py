import logging
from flask import Blueprint, send_from_directory
import config # Import config to get OUTPUT_FOLDER

# Get logger
logger = logging.getLogger(__name__)

# Define Blueprint - Note: No url_prefix needed here as the route is absolute
static_bp = Blueprint('static', __name__)

@static_bp.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves files from the OUTPUT_FOLDER."""
    logger.debug(f"Serving file: {filename} from {config.OUTPUT_FOLDER}")
    # Use config value for the directory
    return send_from_directory(config.OUTPUT_FOLDER, filename, as_attachment=False)