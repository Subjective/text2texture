"""
Flask Backend for Image+Mask-to-3D Generation.

Endpoints:
- /api/generate_image: Generates an image from a text prompt using OpenAI DALL-E.
- /api/predict_mask: Predicts a segmentation mask using SAM2 based on image and points.
- /api/generate_model: Generates a 3D model (STL/PLY) from a heightmap derived from an input image,
                 optionally considering provided masks.
- /outputs/<filename>: Serves generated 3D model files.

Requirements:
- flask
- flask_cors
- torch
- torchvision
- numpy
- opencv-python
- Pillow (PIL)
- requests
- python-dotenv
- openai
- sam2 (install from the cloned repository: pip install -e .)
- A downloaded SAM2 checkpoint (e.g., sam2.1_hiera_large.pt)
- Corresponding SAM2 config file (e.g., configs/sam2.1/sam2.1_hiera_l.yaml)
- heightmap_to_3d.py (library for 3D generation)
- generate_depth.py (library for depth estimation, e.g., using ZoeDepth)
"""
import base64
import io
import logging
import os
import uuid
import requests
import shutil # Added for file copying
import json # For parsing mask metadata

import cv2 # Used for PNG encoding/decoding
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# --- Custom Libraries ---
# Assume these scripts are in the same directory or accessible in the Python path
try:
    from heightmap_to_3d import generate_block_from_heightmap
    from generate_depth import get_grayscale_depth # ZoeDepth-based function
    # Import the refactored function for auto-mask generation
    from autolabel_and_segment import generate_masks_from_image
except ImportError as e:
    print(f"Error importing custom libraries (heightmap_to_3d, generate_depth, autolabel_and_segment): {e}")
    print("Ensure these Python files are accessible.")
    exit(1)

# --- SAM2 Imports (for point-based prediction) ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(
        "Error importing SAM2. Make sure you have cloned the repository"
        " and installed it using 'pip install -e .'"
    )
    print(f"Import Error: {e}")
    # Don't exit immediately, maybe only SAM2 is broken
    # exit(1)

# --- SAM-HQ and Florence-2 Imports (for auto-mask generation) ---
try:
    from segment_anything_hq import sam_model_registry, SamPredictor
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as e:
    print(f"Error importing SAM-HQ/Florence-2 modules: {e}")
    print("Please ensure 'segment-anything-hq' and 'transformers' are installed.")
    # Don't exit immediately, maybe only auto-labeling is broken

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# Folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# SAM2 Paths (Update these paths as needed)
CONFIG_PATH = os.getenv("SAM2_CONFIG_PATH", "configs/sam2.1/sam2.1_hiera_l.yaml") # For SAM2 point prediction
CHECKPOINT_PATH = os.getenv("SAM2_CHECKPOINT_PATH", "checkpoints/sam2.1_hiera_large.pt") # For SAM2 point prediction

# SAM-HQ Paths (Update these paths as needed, loaded from .env)
SAM_HQ_MODEL_TYPE = os.getenv("SAM_HQ_MODEL_TYPE", "vit_h") # e.g., vit_h, vit_l
SAM_HQ_CHECKPOINT_PATH = os.getenv("SAM_HQ_CHECKPOINT_PATH") # Must be set in .env

# Florence-2 Path (Hugging Face model ID)
FLORENCE2_MODEL_ID = os.getenv("FLORENCE2_MODEL_ID", "microsoft/Florence-2-large-ft")

# --- Device Selection (Used by all models) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    # MPS support is experimental for SAM2
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Autocast Dtype (for SAM2) ---
if DEVICE.type == "cuda":
    AUTOCAST_DTYPE = torch.bfloat16
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    AUTOCAST_DTYPE = torch.float32

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}")
logger.info(f"Using SAM2 autocast dtype: {AUTOCAST_DTYPE}")
if DEVICE.type == "mps":
      print(
        "\nWarning: Support for MPS devices is preliminary for SAM2. "
        "It might give numerically different outputs and sometimes degraded performance."
    )

# --- OpenAI Client Setup ---
try:
    # OPENAI_API_KEY should be in .env or environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment or .env file. Text-to-image generation will be disabled.")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=openai_api_key)
        # Optional: Test connection (add error handling if needed)
        # openai_client.models.list()
        logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
    openai_client = None
# -------------------------

# --- Global Variables ---
sam2_predictor: SAM2ImagePredictor | None = None # For point-based SAM2
# Globals for Auto-labeling models
florence_model = None
florence_processor = None
sam_hq_predictor: SamPredictor | None = None


# --- Helper Functions ---

def generate_checkerboard_heightmap(mask_region: np.ndarray, checker_size: int = 8, height: float = 1.0) -> np.ndarray:
    """
    Generates a heightmap with a checkerboard pattern within the mask region.

    Args:
        mask_region: A 2D boolean numpy array where True indicates the region for the checkerboard.
        checker_size: The size of each square in the checkerboard pattern.
        height: The value to assign to the 'high' squares of the checkerboard.

    Returns:
        A 2D numpy array of the same shape as mask_region, with the checkerboard pattern
        applied within the mask, and zeros elsewhere.
    """
    if mask_region.ndim != 2 or mask_region.dtype != bool:
        raise ValueError("mask_region must be a 2D boolean numpy array.")

    h, w = mask_region.shape
    checkerboard = np.zeros((h, w), dtype=np.float32)

    # Create the checkerboard pattern based on coordinates
    # (i // checker_size) % 2 == (j // checker_size) % 2 determines the square color
    for i in range(h):
        for j in range(w):
            if mask_region[i, j]: # Only apply within the mask
                if (i // checker_size) % 2 == (j // checker_size) % 2:
                    checkerboard[i, j] = height # Or 0, depending on starting corner preference
                else:
                    checkerboard[i, j] = 0 # Or height

    # Alternative vectorized approach (potentially faster for large images)
    # y_idx, x_idx = np.indices(mask_region.shape)
    # checkerboard_pattern = ((y_idx // checker_size) % 2 == (x_idx // checker_size) % 2).astype(np.float32) * height
    # checkerboard[mask_region] = checkerboard_pattern[mask_region]

    return checkerboard


# -- Image Decoding/Encoding --
def decode_base64_image(base64_string: str) -> np.ndarray | None:
    """Decodes a Base64 string into an NumPy array (RGB format)."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Error decoding Base64 image: {e}", exc_info=True)
        return None

def encode_mask_to_base64_png(mask: np.ndarray) -> str | None:
    """Encodes a boolean mask numpy array to a Base64 PNG string with transparency."""
    if mask.ndim != 2:
        logger.error(f"Mask must be 2D, but got shape {mask.shape}")
        return None
    try:
        mask_bool = mask.astype(bool)
        h, w = mask_bool.shape
        # Create a 4-channel BGRA image (OpenCV uses BGRA)
        bgra_mask = np.zeros((h, w, 4), dtype=np.uint8) # Transparent background
        # Set foreground pixels to a visible color (e.g., blue) and opaque
        bgra_mask[mask_bool] = [255, 0, 0, 255] # Blue: B=255, G=0, R=0, Alpha=255

        # Encode the BGRA mask to PNG format in memory
        success, buffer = cv2.imencode('.png', bgra_mask)
        if not success:
            logger.error("Failed to encode mask to PNG format.")
            return None

        # Encode the PNG buffer to Base64 string
        png_base64 = base64.b64encode(buffer).decode('utf-8')
        return png_base64
    except Exception as e:
        logger.error(f"Error encoding mask to Base64 PNG: {e}", exc_info=True)
        return None

# -- Text-to-Image Generation --
def generate_and_save_image(prompt: str, upload_folder: str) -> tuple[str | None, str | None]:
    """Generates image using OpenAI DALL-E and saves it."""
    if not openai_client:
        logger.error("OpenAI client not available for image generation.")
        raise ConnectionError("OpenAI client not initialized. Check API key.")

    logger.info(f"Requesting image generation from OpenAI for prompt: '{prompt}'")
    try:
        response = openai_client.images.generate(
            model="dall-e-3", # Or "dall-e-2" if preferred/available
            prompt=prompt,
            size="1024x1024", # DALL-E 3 requires specific sizes like 1024x1024
            quality="standard", # Or "hd"
            n=1,
            response_format="url" # Get URL to download the image
        )

        image_url = response.data[0].url
        if not image_url:
             raise ValueError("OpenAI response did not contain an image URL.")
        logger.info(f"Received image URL: {image_url}")

        # Download the image
        image_response = requests.get(image_url, stream=True, timeout=60) # Added timeout
        image_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Create a unique filename
        image_filename = f"generated_{str(uuid.uuid4())}.png" # Assume PNG format
        image_path = os.path.join(upload_folder, image_filename)

        # Save the image
        logger.info(f"Downloading and saving generated image to: {image_path}")
        with open(image_path, 'wb') as f:
            for chunk in image_response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Image saved successfully.")
        return image_path, image_filename # Return full path and base filename

    except requests.exceptions.RequestException as e:
         logger.error(f"Network error downloading image from OpenAI URL {image_url}: {e}", exc_info=True)
         raise ConnectionError(f"Failed to download image from OpenAI: {e}")
    except Exception as e:
        # Catch other potential errors (OpenAI API errors, file system errors)
        logger.error(f"Error generating/saving image from prompt '{prompt}': {e}", exc_info=True)
        # Re-raise a more specific or generic error for the main endpoint to catch
        raise ValueError(f"Failed to generate or save image from OpenAI: {e}")

# --- Initialization ---
def load_sam2_model():
    """Loads the SAM2 model predictor."""
    global sam2_predictor
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"SAM2 Config file not found: {CONFIG_PATH}")
        return False
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"SAM2 Checkpoint file not found: {CHECKPOINT_PATH}")
        return False
    try:
        logger.info(f"Building SAM2 base model from {CONFIG_PATH} and {CHECKPOINT_PATH}...")
        sam2_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        logger.info("Initializing SAM2ImagePredictor...")
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"SAM2 predictor loaded successfully on device: {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Error loading SAM2 model: {e}", exc_info=True)
        sam2_predictor = None
        return False

# --- Autolabel Model Loading ---
def load_autolabel_models():
    """Loads the Florence-2 and SAM-HQ models."""
    global florence_model, florence_processor, sam_hq_predictor

    # --- Load Florence-2 ---
    try:
        logger.info(f"Loading Florence-2 model: {FLORENCE2_MODEL_ID}...")
        # Load model directly to the determined device
        florence_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
        florence_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
        logger.info("Florence-2 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Florence-2 model '{FLORENCE2_MODEL_ID}': {e}", exc_info=True)
        logger.error("Ensure the model ID is correct, transformers/accelerate are installed, and you have internet connectivity.")
        florence_model = None # Mark as unloaded
        florence_processor = None
        # Continue loading other models if possible

    # --- Load SAM-HQ ---
    if not SAM_HQ_CHECKPOINT_PATH or not os.path.exists(SAM_HQ_CHECKPOINT_PATH):
        logger.error(f"SAM-HQ checkpoint file not found or not specified in .env (SAM_HQ_CHECKPOINT_PATH). Path: '{SAM_HQ_CHECKPOINT_PATH}'")
        sam_hq_predictor = None # Mark as unloaded
        return # Cannot proceed without checkpoint

    try:
        logger.info(f"Loading SAM-HQ model type: {SAM_HQ_MODEL_TYPE} structure...")
        # Instantiate model structure (try checkpoint=None first)
        sam_model_hq = sam_model_registry[SAM_HQ_MODEL_TYPE](checkpoint=None)

        logger.info(f"Loading SAM-HQ checkpoint state_dict: {SAM_HQ_CHECKPOINT_PATH} with map_location='{DEVICE}'...")
        state_dict = torch.load(SAM_HQ_CHECKPOINT_PATH, map_location=DEVICE)
        sam_model_hq.load_state_dict(state_dict)
        sam_model_hq = sam_model_hq.to(DEVICE).eval()
        sam_hq_predictor = SamPredictor(sam_model_hq)
        logger.info(f"SAM-HQ predictor ('{SAM_HQ_MODEL_TYPE}') loaded successfully on device: {DEVICE}")

    except TypeError as te:
         if "'NoneType'" in str(te):
              logger.warning(f"Failed loading SAM-HQ structure with checkpoint=None ({te}). Attempting workaround...")
              try:
                  # WORKAROUND: Instantiate with the checkpoint path, then immediately overwrite state_dict
                  sam_model_hq = sam_model_registry[SAM_HQ_MODEL_TYPE](checkpoint=SAM_HQ_CHECKPOINT_PATH) # Initial load
                  state_dict = torch.load(SAM_HQ_CHECKPOINT_PATH, map_location=DEVICE) # Load correctly mapped state_dict
                  sam_model_hq.load_state_dict(state_dict) # Overwrite
                  sam_model_hq = sam_model_hq.to(DEVICE).eval() # Ensure final move and eval mode
                  sam_hq_predictor = SamPredictor(sam_model_hq)
                  logger.info(f"SAM-HQ predictor ('{SAM_HQ_MODEL_TYPE}') loaded successfully using workaround.")
              except Exception as e_workaround:
                  logger.error(f"SAM-HQ workaround failed: {e_workaround}", exc_info=True)
                  sam_hq_predictor = None # Mark as unloaded
         else:
            logger.error(f"Error loading SAM-HQ model (TypeError): {te}", exc_info=True)
            sam_hq_predictor = None # Mark as unloaded
    except KeyError:
         logger.error(f"Error: SAM-HQ model type '{SAM_HQ_MODEL_TYPE}' not found in registry.")
         sam_hq_predictor = None # Mark as unloaded
    except Exception as e:
        logger.error(f"Error loading SAM-HQ model: {e}", exc_info=True)
        sam_hq_predictor = None # Mark as unloaded

# --- Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend requests

# --- API Endpoints ---

@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    """API endpoint to generate an image from a text prompt."""
    if not openai_client:
        return jsonify({"error": "Image generation service is not configured or unavailable."}), 503 # Service Unavailable

    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request data"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "'prompt' cannot be empty"}), 400

    try:
        # Generate and save the image, get its path
        image_path, _ = generate_and_save_image(prompt, UPLOAD_FOLDER)

        if not image_path: # Should be caught by exception, but double-check
             return jsonify({"error": "Failed to generate image path."}), 500

        # Read the saved image and encode it to Base64 to send back
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up the generated file immediately after reading? Optional.
        # try:
        #     os.remove(image_path)
        #     logger.info(f"Cleaned up generated image file: {image_path}")
        # except OSError as e:
        #     logger.warning(f"Could not clean up generated image file {image_path}: {e}")

        return jsonify({
            "status": "success",
            "image_b64": img_base64 # Send base64 data back
        })

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"Error in /api/generate_image endpoint: {e}", exc_info=True)
        # Return a user-friendly error
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/api/predict_mask", methods=["POST"])
def api_predict_mask():
    """API endpoint to predict mask based on image and points using SAM2."""
    global sam2_predictor
    if sam2_predictor is None:
        logger.error("SAM2 model not loaded, cannot predict mask.")
        return jsonify({"status": "error", "message": "Mask prediction model not loaded"}), 503 # Service Unavailable

    data = request.get_json()
    if not data: return jsonify({"status": "error", "message": "No input data provided"}), 400

    base64_image_str = data.get("image")
    points_data = data.get("points")

    if not base64_image_str or not points_data:
        return jsonify({"status": "error", "message": "Missing 'image' or 'points' in request"}), 400
    if not isinstance(points_data, list) or len(points_data) == 0:
        return jsonify({"status": "error", "message": "'points' must be a non-empty list of {x, y} objects"}), 400

    logger.info("Decoding image for mask prediction...")
    image_rgb = decode_base64_image(base64_image_str)
    if image_rgb is None: return jsonify({"status": "error", "message": "Invalid image data provided"}), 400
    logger.info(f"Image decoded successfully: shape={image_rgb.shape}")

    try:
        # Convert points from frontend format [{x: val, y: val}, ...] to numpy array
        point_coords_np = np.array([[p["x"], p["y"]] for p in points_data])
        # SAM2 expects labels (1 for foreground point)
        point_labels_np = np.ones(point_coords_np.shape[0])
        # SAM2 expects batch dimension: (1, num_points, 2) and (1, num_points)
        point_coords = point_coords_np[None, :, :]
        point_labels = point_labels_np[None, :]
        logger.info(f"Processed {point_coords.shape[1]} points. Input shape: {point_coords.shape}")
    except (KeyError, TypeError, IndexError) as e:
        logger.error(f"Invalid point format received: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Invalid point format: {e}"}), 400

    # --- Run SAM2 Prediction ---
    try:
        with torch.autocast(device_type=DEVICE.type, dtype=AUTOCAST_DTYPE):
            logger.info("Setting image in SAM2 predictor...")
            sam2_predictor.set_image(image_rgb) # Set the image for the predictor
            logger.info("Running SAM2 prediction...")
            # multimask_output=False returns the highest-scoring mask
            masks, scores, logits = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False, # Get single best mask
            )
            logger.info(f"Prediction complete. Mask type: {type(masks)}, Scores type: {type(scores)}")

            # Process the output mask (should be shape [1, H, W])
            if isinstance(masks, torch.Tensor):
                if masks.ndim == 3 and masks.shape[0] == 1:
                    output_mask_tensor = masks.squeeze(0) # Remove batch dim -> [H, W]
                    output_mask = output_mask_tensor.cpu().numpy() # To numpy
                else:
                     raise ValueError(f"Unexpected mask tensor output shape: {masks.shape}. Expected (1, H, W).")
            elif isinstance(masks, np.ndarray):
                 if masks.ndim == 3 and masks.shape[0] == 1:
                     output_mask = masks.squeeze(0) # Remove batch dim -> [H, W]
                 else:
                     raise ValueError(f"Unexpected mask numpy output shape: {masks.shape}. Expected (1, H, W).")
            else:
                raise TypeError(f"Unexpected mask type from predictor: {type(masks)}")

            output_mask_bool = output_mask.astype(bool) # Ensure boolean type

    except Exception as e:
        logger.error(f"Error during SAM2 prediction: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Mask prediction failed: {e}"}), 500

    # --- Encode Result Mask to Base64 PNG ---
    logger.info("Encoding predicted mask to Base64 PNG...")
    b64_png_mask = encode_mask_to_base64_png(output_mask_bool)
    if b64_png_mask is None:
        return jsonify({"status": "error", "message": "Failed to encode predicted mask to PNG"}), 500
    logger.info("Mask Base64 PNG encoding successful.")

    # --- Extract Score ---
    score = None
    if scores is not None:
        try:
            # Scores tensor might be shape [1] or similar
            if isinstance(scores, torch.Tensor): score_scalar = scores.squeeze(); score = score_scalar.cpu().item() if score_scalar.numel() == 1 else None
            elif isinstance(scores, np.ndarray): score_scalar = scores.squeeze(); score = score_scalar.item() if score_scalar.size == 1 else None
            if score is None: logger.warning(f"Could not extract single score. Shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
        except Exception as score_err: logger.warning(f"Could not extract single score: {score_err}. Type: {type(scores)}")

    # --- Return Response ---
    return jsonify({
        "status": "success",
        "mask_b64png": b64_png_mask, # Key matches frontend expectation
        "score": score,
    })

@app.route("/api/generate_masks", methods=["POST"])
def api_generate_masks():
    """API endpoint for automatic mask generation using Florence-2 and SAM-HQ."""
    global florence_model, florence_processor, sam_hq_predictor

    if not all([florence_model, florence_processor, sam_hq_predictor]):
        logger.error("Auto-labeling models not loaded, cannot generate masks.")
        return jsonify({"status": "error", "message": "Automatic mask generation service not available"}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "Missing 'image' in request data"}), 400

    base64_image_str = data["image"]
    if not base64_image_str:
        return jsonify({"status": "error", "message": "'image' cannot be empty"}), 400

    logger.info("Decoding image for automatic mask generation...")
    image_rgb = decode_base64_image(base64_image_str)
    if image_rgb is None:
        return jsonify({"status": "error", "message": "Invalid image data provided"}), 400
    logger.info(f"Image decoded successfully: shape={image_rgb.shape}")

    try:
        logger.info("Calling generate_masks_from_image function...")
        # Call the refactored function with pre-loaded models
        # Assuming default hq_token_only=False for now, could make this configurable via request
        masks_tensor, labels, scores_tensor = generate_masks_from_image(
            image_rgb=image_rgb,
            florence_model=florence_model,
            florence_processor=florence_processor,
            sam_predictor=sam_hq_predictor,
            device=DEVICE,
            hq_token_only=False # Or get from request if needed
        )
        logger.info(f"Received {masks_tensor.shape[0]} masks from generation function.")

        # Process results
        results = []
        # Move tensors to CPU for iteration and encoding
        masks_cpu = masks_tensor.cpu()
        scores_cpu = scores_tensor.cpu().tolist() # Convert scores to list

        for i in range(masks_cpu.shape[0]):
            mask_bool = masks_cpu[i].numpy().astype(bool) # Get individual mask as numpy bool array
            label = labels[i] if i < len(labels) else "unknown"
            score = scores_cpu[i] if i < len(scores_cpu) else 0.0

            logger.debug(f"Encoding mask {i} ('{label}')...")
            b64_png_mask = encode_mask_to_base64_png(mask_bool)
            if b64_png_mask:
                results.append({
                    "mask_b64png": b64_png_mask, # Matches frontend AutoGeneratedMaskData type
                    "label": label,
                    "score": score,
                })
            else:
                logger.warning(f"Failed to encode mask {i} ('{label}') to Base64 PNG.")

        logger.info(f"Successfully processed and encoded {len(results)} masks.")
        return jsonify({
            "status": "success",
            "masks": results # Return the list of mask objects
        })

    except Exception as e:
        logger.error(f"Error during automatic mask generation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Automatic mask generation failed: {e}"}), 500


@app.route("/api/generate_model", methods=["POST"])
def api_generate_3d_model():
    """
    API endpoint to generate the final 3D model (STL/PLY).
    Accepts the original image, optional masks, and 3D parameters.
    """
    color_image_path = None
    color_image_filename = None
    saved_mask_files = {} # Dictionary to store paths to saved mask files { 'mask_0': path, ... }
    mask_metadata = [] # List to store metadata [{id:.., name:..}, ...]
    heightmap_path = None # Path to the base heightmap
    merged_heightmap_path = None # Path to the heightmap with details merged
    files_to_remove = [] # Keep track of files for cleanup

    # --- 1) Handle Input Image (Upload) ---
    # Expecting 'color_image' file part from the frontend's FormData
    if "color_image" not in request.files or request.files["color_image"].filename == '':
         return jsonify({"error": "Missing 'color_image' file in request"}), 400

    color_image_file = request.files["color_image"]
    # Use a UUID to ensure unique filename in uploads folder
    color_image_filename = f"color_{str(uuid.uuid4())}_{color_image_file.filename}"
    color_image_path = os.path.join(UPLOAD_FOLDER, color_image_filename)
    files_to_remove.append(color_image_path) # Add to cleanup list immediately
    try:
        color_image_file.save(color_image_path)
        logger.info(f"Saved uploaded color image to: {color_image_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded color_image file: {str(e)}", exc_info=True)
        # TODO: Cleanup already saved files before returning
        return jsonify({"error": f"Could not save uploaded file: {str(e)}"}), 500

    # --- 2) Handle Input Masks (Optional) ---
    # Expecting 'mask_0', 'mask_1', ... file parts and 'mask_metadata' form field
    mask_files = request.files.getlist('mask') # Or use a pattern if names are 'mask_0', 'mask_1' etc.
    # Alternative: Iterate through all files checking for a pattern
    for key in request.files:
        if key.startswith("mask_"):
             mask_file = request.files[key]
             if mask_file and mask_file.filename:
                 mask_filename = f"mask_{str(uuid.uuid4())}_{mask_file.filename}"
                 mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                 files_to_remove.append(mask_path) # Add to cleanup list
                 try:
                     mask_file.save(mask_path)
                     # Store the path using the original key ('mask_0', 'mask_1', etc.)
                     saved_mask_files[key] = mask_path
                     logger.info(f"Saved uploaded mask '{key}' to: {mask_path}")
                 except Exception as e:
                     logger.error(f"Error saving uploaded mask file '{key}': {str(e)}", exc_info=True)
                     # TODO: Clean up already saved files (color image, other masks)
                     return jsonify({"error": f"Could not save mask file '{key}': {str(e)}"}), 500

    # Get mask metadata (e.g., names, IDs) sent as JSON string in form data
    # Ensure metadata aligns with saved_mask_files if possible (e.g., sort keys)
    mask_keys_sorted = sorted(saved_mask_files.keys()) # e.g., ['mask_0', 'mask_1', ...]
    if "mask_metadata" in request.form:
        try:
            mask_metadata = json.loads(request.form["mask_metadata"])
            if not isinstance(mask_metadata, list):
                 logger.warning("Received mask_metadata is not a list.")
                 mask_metadata = [] # Reset if format is wrong
            else:
                 logger.info(f"Received metadata for {len(mask_metadata)} masks.")
        except json.JSONDecodeError:
            logger.warning("Failed to decode mask_metadata JSON string.")
            mask_metadata = [] # Reset on decode error

    logger.info(f"Processed {len(saved_mask_files)} mask files.")

    # --- 3) Generate Heightmap (from color image) ---
    # Create a unique filename for the heightmap
    unique_base_heightmap_filename = f"base_depth_{str(uuid.uuid4())}.png"
    heightmap_path = os.path.join(UPLOAD_FOLDER, unique_base_heightmap_filename)
    files_to_remove.append(heightmap_path) # Add base heightmap to cleanup

    logger.info(f"Generating base heightmap for {color_image_path} -> {heightmap_path}")
    try:
        # Call the depth generation function (e.g., using ZoeDepth)
        get_grayscale_depth(color_image_path, heightmap_path)
        logger.info(f"Successfully generated base heightmap: {heightmap_path}")
        # Load the base heightmap
        base_heightmap_np = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
        if base_heightmap_np is None:
            raise ValueError(f"Failed to load generated base heightmap from {heightmap_path}")
        # Normalize base heightmap (optional, depends on generate_depth output)
        # base_heightmap_np = base_heightmap_np.astype(np.float32) / 255.0
        base_heightmap_np = base_heightmap_np.astype(np.float32) # Ensure float for merging
        final_heightmap_np = base_heightmap_np.copy() # Start final map with base
        logger.info(f"Base heightmap loaded, shape: {base_heightmap_np.shape}")

    except Exception as e:
        logger.error(f"Error generating or loading base heightmap for {color_image_path}: {str(e)}", exc_info=True)
        # TODO: Clean up uploaded files
        return jsonify({"error": f"Error generating base heightmap: {str(e)}"}), 500

    # --- 4) Merge Mask Details onto Heightmap ---
    if saved_mask_files:
        logger.info(f"Merging details for {len(saved_mask_files)} masks...")
        detail_scale = 50.0 # Scaling factor for checkerboard height
        checker_size = 8 # Size for checkerboard squares

        # Iterate through masks in the order defined by sorted keys
        for mask_key in mask_keys_sorted:
            mask_path = saved_mask_files[mask_key]
            try:
                # Load the mask (assuming it's a single channel PNG where non-zero is the mask)
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    logger.warning(f"Could not load mask file {mask_path}, skipping detail.")
                    continue
                # Ensure mask is boolean and matches heightmap shape
                if mask_img.shape != final_heightmap_np.shape:
                     logger.warning(f"Mask {mask_path} shape {mask_img.shape} mismatch with heightmap {final_heightmap_np.shape}, resizing mask...")
                     mask_img = cv2.resize(mask_img, (final_heightmap_np.shape[1], final_heightmap_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                current_mask_bool = mask_img.astype(bool)

                # Generate checkerboard detail for this mask
                checkerboard_detail_np = generate_checkerboard_heightmap(
                    current_mask_bool,
                    checker_size=checker_size,
                    height=1.0 # Generate checkerboard with height 1, scale later
                )

                # Add scaled detail to the final heightmap within the mask region
                final_heightmap_np[current_mask_bool] += detail_scale * checkerboard_detail_np[current_mask_bool]
                logger.debug(f"Applied checkerboard detail for mask {mask_key}")

            except Exception as e:
                logger.error(f"Error processing mask {mask_path} for detail merging: {e}", exc_info=True)
                # Decide whether to continue or fail
                # return jsonify({"error": f"Error processing mask {mask_key}: {e}"}), 500

        # Normalize the final heightmap (optional, but good practice if values might exceed range)
        min_h, max_h = np.min(final_heightmap_np), np.max(final_heightmap_np)
        if max_h > min_h:
             final_heightmap_np = (final_heightmap_np - min_h) / (max_h - min_h)
        # Convert back to uint8 if needed by the 3D generation library, otherwise keep float
        # final_heightmap_np = (final_heightmap_np * 255).astype(np.uint8)

        # Save the final merged heightmap
        merged_heightmap_filename = f"merged_depth_{str(uuid.uuid4())}.png"
        merged_heightmap_path = os.path.join(UPLOAD_FOLDER, merged_heightmap_filename)
        files_to_remove.append(merged_heightmap_path) # Add merged map to cleanup
        try:
            # Save as float image if possible (e.g., TIFF) or scale to uint8 PNG
            # Saving as uint8 PNG for compatibility
            final_heightmap_uint8 = (final_heightmap_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(merged_heightmap_path, final_heightmap_uint8)
            logger.info(f"Saved final merged heightmap to: {merged_heightmap_path}")
            heightmap_to_use = merged_heightmap_path # Use this path for 3D generation
        except Exception as e:
             logger.error(f"Error saving merged heightmap: {e}", exc_info=True)
             # TODO: Cleanup
             return jsonify({"error": f"Error saving merged heightmap: {e}"}), 500
    else:
        logger.info("No masks provided, using base heightmap for 3D generation.")
        heightmap_to_use = heightmap_path # Use the original base heightmap path

    # --- 5) Retrieve 3D Model Parameters ---
    try:
        block_width = float(request.form.get("block_width", 100))
        block_length = float(request.form.get("block_length", 100))
        block_thickness = float(request.form.get("block_thickness", 10))
        depth = float(request.form.get("depth", 5)) # Max height/depth variation
        base_height = float(request.form.get("base_height", 0))
        mode = request.form.get("mode", "protrude") # 'protrude' or 'carve'
        invert = request.form.get("invert", "false").lower() == "true"
        include_color = request.form.get("include_color", "false").lower() == "true"
        logger.info(f"Retrieved 3D parameters: mode={mode}, invert={invert}, color={include_color}")
    except ValueError:
        logger.warning("Invalid numeric parameters received for 3D generation.")
        # TODO: Clean up uploaded files
        return jsonify({"error": "Invalid 3D parameter values"}), 400

    # --- 6) Set Output File Type and Path ---
    if include_color:
        file_type = "ply"
        color_reference_param = color_image_path # Use the path of the uploaded color image
        output_filename = str(uuid.uuid4()) + ".ply"
    else:
        file_type = "stl"
        color_reference_param = None
        output_filename = str(uuid.uuid4()) + ".stl"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    output_heightmap_filename = os.path.splitext(output_filename)[0] + "_heightmap.png"
    output_heightmap_path = os.path.join(OUTPUT_FOLDER, output_heightmap_filename)
    logger.info(f"Preparing to generate {file_type.upper()} model. Output: {output_path}")

    # --- 7) Generate the 3D Model ---
    try:
        logger.info(f"Generating 3D model using heightmap: {heightmap_to_use}")
        generate_block_from_heightmap(
            heightmap_path=heightmap_to_use, # Use the final (potentially merged) heightmap
            output_path=output_path,
            block_width=block_width,
            block_length=block_length,
            block_thickness=block_thickness,
            depth=depth,
            base_height=base_height,
            mode=mode,
            invert=invert,
            color_reference=color_reference_param,
            # --------------------------------------------------------
        )
        logger.info(f"Successfully generated 3D model: {output_path}")
    except TypeError as te:
         # Catch if the function doesn't accept the new mask arguments yet
         if 'mask_paths' in str(te) or 'mask_metadata' in str(te):
              logger.error(f"Mismatch calling generation function. Does `generate_block_from_heightmap` accept 'mask_paths' and 'mask_metadata' arguments? Error: {te}", exc_info=True)
              # TODO: Clean up files
              return jsonify({"error": "Internal configuration error: 3D generation function needs update for masks."}), 500
         else:
              logger.error(f"Type error during 3D model generation: {str(te)}", exc_info=True)
              # TODO: Clean up files
              return jsonify({"error": f"Type error during 3D model generation: {str(te)}"}), 500
    except Exception as e:
        logger.error(f"Error during 3D model generation: {str(e)}", exc_info=True)
        # TODO: Clean up uploaded/intermediate files
        return jsonify({"error": f"Error generating 3D model: {str(e)}"}), 500

    # --- 8) Copy final heightmap to output and Clean up temporary files ---
    try:
        logger.info(f"Copying final heightmap {heightmap_to_use} to {output_heightmap_path}")
        shutil.copy2(heightmap_to_use, output_heightmap_path) # Use copy2 to preserve metadata if possible
        # Now remove the original heightmap paths from the cleanup list if they exist
        if heightmap_path in files_to_remove:
            files_to_remove.remove(heightmap_path)
            logger.debug(f"Removed base heightmap {heightmap_path} from cleanup list.")
        if merged_heightmap_path and merged_heightmap_path in files_to_remove:
            files_to_remove.remove(merged_heightmap_path)
            logger.debug(f"Removed merged heightmap {merged_heightmap_path} from cleanup list.")
    except Exception as e:
        logger.error(f"Error copying final heightmap to output folder: {e}", exc_info=True)
        # Proceed with cleanup, but the heightmap download might fail

    # files_to_remove list now contains all intermediate files EXCEPT the final heightmap
    logger.info(f"Cleaning up {len(files_to_remove)} temporary files...")
    for f_path in files_to_remove:
        try:
            if f_path and os.path.exists(f_path):
                os.remove(f_path)
                logger.debug(f"Cleaned up temporary file: {f_path}") # Changed to debug
        except OSError as e:
            logger.warning(f"Could not clean up temporary file {f_path}: {e}")

    # --- 9) Return Success Response ---
    # Construct the URL for the frontend to fetch the generated file
    file_url = request.host_url.rstrip('/') + "/outputs/" + output_filename
    heightmap_file_url = request.host_url.rstrip('/') + "/outputs/" + output_heightmap_filename
    logger.info(f"Returning success response. Model URL: {file_url}, Heightmap URL: {heightmap_file_url}")
    return jsonify({
        "fileUrl": file_url,
        "fileType": file_type,
        "heightmapFileUrl": heightmap_file_url # Add the heightmap URL
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves files from the OUTPUT_FOLDER."""
    logger.debug(f"Serving file: {filename} from {OUTPUT_FOLDER}")
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False) # Send inline if possible

# --- Main Execution ---
if __name__ == "__main__":
    # Determine host, port, and debug mode first
    host = os.getenv("FLASK_HOST", "127.0.0.1") # Default to 127.0.0.1 (localhost)
    port = int(os.getenv("FLASK_PORT", 5000)) # Default to 5000
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]

    # Check if we are in the reloader's main process (child process)
    # Only load models in the actual worker process to avoid double loading
    # WERKZEUG_RUN_MAIN is set by Flask/Werkzeug when using the reloader
    is_reloader_main_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    if not debug_mode or is_reloader_main_process:
        logger.info("Loading models...")
        # Load SAM2 model (for point prediction)
        if not load_sam2_model():
            logger.warning("Failed to load SAM2 model. Point-based mask prediction endpoint will not work.")
            # Decide if the app should exit or run without the model
            # exit(1) # Uncomment to exit if SAM2 loading fails

        # Load Autolabel models (Florence-2 + SAM-HQ)
        load_autolabel_models() # This function logs errors internally
        if not all([florence_model, florence_processor, sam_hq_predictor]):
             logger.warning("One or more auto-labeling models failed to load. Automatic mask generation endpoint will not work.")
             # Decide if the app should exit or run without these models
        logger.info("Model loading complete (or attempted).")
    else:
        logger.info("Skipping model loading in Flask reloader parent process.")


    # Start Flask development server
    # The app.run() call needs to happen regardless of the process type
    # for the reloader mechanism to function correctly.
    logger.info(f"Starting Flask server on {host}:{port} (Debug: {debug_mode})...")
    # Let app.run handle the reloader based on debug_mode; don't set use_reloader=False
    app.run(host=host, port=port, debug=debug_mode)
    logger.info("Flask backend stopped.")
