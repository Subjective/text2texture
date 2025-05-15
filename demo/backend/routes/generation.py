import base64
import logging
import os
import uuid
import requests
import shutil
import json
import cv2
import numpy as np
from PIL import Image

from flask import Blueprint, jsonify, request

# Import configuration, models, processing functions, and utilities
import config
import models
from processing.depth import get_grayscale_depth
from processing.conversion_3d import generate_block_from_heightmap
from utils.image_utils import generate_heightmap_by_texture_type

# Get logger
logger = logging.getLogger(__name__)

# Define Blueprint
generation_bp = Blueprint('generation', __name__, url_prefix='/api')

# --- Helper Function ---

def generate_and_save_image(prompt: str, upload_folder: str) -> tuple[str | None, str | None]:
    """Generates image using OpenAI DALL-E and saves it."""
    openai_client = models.get_openai_client() # Get client from models module
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
        image_response = requests.get(image_url, stream=True, timeout=60)
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


# --- API Endpoints ---

@generation_bp.route("/generate_image", methods=["POST"])
def api_generate_image():
    """API endpoint to generate an image from a text prompt."""
    openai_client = models.get_openai_client() # Check if client is available
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
        # Use config for upload folder
        image_path, _ = generate_and_save_image(prompt, config.UPLOAD_FOLDER)

        if not image_path: # Should be caught by exception, but double-check
             return jsonify({"error": "Failed to generate image path."}), 500

        # Read the saved image and encode it to Base64 to send back
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up the generated file immediately after reading?
        try:
            os.remove(image_path)
            logger.info(f"Cleaned up generated image file: {image_path}")
        except OSError as e:
            logger.warning(f"Could not clean up generated image file {image_path}: {e}")

        return jsonify({
            "status": "success",
            "image_b64": img_base64 # Send base64 data back
        })

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"Error in /api/generate_image endpoint: {e}", exc_info=True)
        # Return a user-friendly error
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@generation_bp.route("/generate_model", methods=["POST"])
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
    if "color_image" not in request.files or request.files["color_image"].filename == '':
         return jsonify({"error": "Missing 'color_image' file in request"}), 400

    color_image_file = request.files["color_image"]
    # Sanitize filename (though uuid makes it unique, good practice for the original part)
    # from werkzeug.utils import secure_filename
    # original_filename = secure_filename(color_image_file.filename)
    # color_image_filename = f"color_{str(uuid.uuid4())}_{original_filename}"
    color_image_filename = f"color_{str(uuid.uuid4())}_{color_image_file.filename}" # Simpler for now

    color_image_path = os.path.join(config.UPLOAD_FOLDER, color_image_filename)
    files_to_remove.append(color_image_path)
    pil_color_image = None # For storing the loaded PIL image

    try:
        color_image_file.save(color_image_path)
        logger.info(f"Saved uploaded color image to: {color_image_path}")
        # Load the image into PIL after saving
        pil_color_image = Image.open(color_image_path)
        # pil_color_image.load() # Ensure it's fully loaded, usually not needed with open()
        logger.info(f"Loaded color image into PIL Image object.")
    except FileNotFoundError:
        logger.error(f"Failed to find saved image at {color_image_path} for PIL loading.", exc_info=True)
        return jsonify({"error": "Failed to process uploaded image after saving."}), 500
    except Exception as e:
        logger.error(f"Error saving or loading uploaded color_image file: {str(e)}", exc_info=True)
        # TODO: Cleanup already saved files before returning
        return jsonify({"error": f"Could not save or load uploaded file: {str(e)}"}), 500
    
    if pil_color_image is None: # Should be caught by exceptions above, but as a safeguard
        logger.error("PIL color image was not loaded despite no explicit exception.")
        return jsonify({"error": "Failed to load color image for processing."}), 500


    # --- 2) Handle Input Masks (Optional) ---
    for key in request.files:
        if key.startswith("mask_"):
             mask_file = request.files[key]
             if mask_file and mask_file.filename:
                 mask_filename = f"mask_{str(uuid.uuid4())}_{mask_file.filename}"
                 mask_path = os.path.join(config.UPLOAD_FOLDER, mask_filename)
                 files_to_remove.append(mask_path)
                 try:
                     mask_file.save(mask_path)
                     saved_mask_files[key] = mask_path
                     logger.info(f"Saved uploaded mask '{key}' to: {mask_path}")
                 except Exception as e:
                     logger.error(f"Error saving uploaded mask file '{key}': {str(e)}", exc_info=True)
                     # TODO: Clean up already saved files
                     return jsonify({"error": f"Could not save mask file '{key}': {str(e)}"}), 500

    mask_keys_sorted = sorted(saved_mask_files.keys())
    if "mask_metadata" in request.form:
        try:
            mask_metadata = json.loads(request.form["mask_metadata"])
            if not isinstance(mask_metadata, list):
                 logger.warning("Received mask_metadata is not a list.")
                 mask_metadata = []
            else:
                 logger.info(f"Received metadata for {len(mask_metadata)} masks.")
        except json.JSONDecodeError:
            logger.warning("Failed to decode mask_metadata JSON string.")
            mask_metadata = []

    logger.info(f"Processed {len(saved_mask_files)} mask files.")

    # --- 3) Generate Heightmap (from color image) ---
    unique_base_heightmap_filename = f"base_depth_{str(uuid.uuid4())}.png"
    heightmap_path = os.path.join(config.UPLOAD_FOLDER, unique_base_heightmap_filename)
    files_to_remove.append(heightmap_path)

    logger.info(f"Generating base heightmap using PIL image -> {heightmap_path}")
    try:
        # Use imported function with the PIL image
        base_heightmap_np_from_depth_func = get_grayscale_depth(pil_color_image) # Returns NumPy array
        
        # Save the returned NumPy array as an image
        Image.fromarray(base_heightmap_np_from_depth_func).save(heightmap_path)
        logger.info(f"Successfully generated and saved base heightmap: {heightmap_path}")

        # The returned array is already the processed depth map (inverted, grayscale, uint8)
        # For consistency with subsequent processing that uses cv2.imread and float32:
        base_heightmap_np = base_heightmap_np_from_depth_func.astype(np.float32) 

        final_heightmap_np = base_heightmap_np.copy()
        logger.info(f"Base heightmap processed, shape: {base_heightmap_np.shape}, type: {base_heightmap_np.dtype}")

    except Exception as e:
        logger.error(f"Error generating or saving base heightmap from PIL image: {str(e)}", exc_info=True)
        # TODO: Clean up uploaded files
        return jsonify({"error": f"Error generating base heightmap: {str(e)}"}), 500

    # --- 4) Merge Mask Details onto Heightmap ---
    if saved_mask_files:
        logger.info(f"Merging details for {len(saved_mask_files)} masks...")
        detail_scale = 50.0
        feature_size = 8

        for mask_key in mask_keys_sorted:
            mask_path = saved_mask_files[mask_key]
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    logger.warning(f"Could not load mask file {mask_path}, skipping detail.")
                    continue
                if mask_img.shape != final_heightmap_np.shape:
                     logger.warning(f"Mask {mask_path} shape {mask_img.shape} mismatch with heightmap {final_heightmap_np.shape}, resizing mask...")
                     mask_img = cv2.resize(mask_img, (final_heightmap_np.shape[1], final_heightmap_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                current_mask_bool = mask_img.astype(bool)
                
                # Find the corresponding metadata for this mask
                mask_id = None
                texture_type = "checkerboard"  # Default texture type
                
                # Extract the mask number from the key (e.g., "mask_0" -> "0")
                mask_num = mask_key.split('_')[1]
                
                # Find metadata for this mask
                for meta in mask_metadata:
                    if meta.get("maskKey") == mask_key or meta.get("id", "").endswith(mask_num):
                        texture_type = meta.get("textureType", "checkerboard")
                        break
                
                image_param_for_texture_type = None
                if texture_type == "auto":
                    if pil_color_image: # We have the PIL image loaded already
                        image_param_for_texture_type = pil_color_image # Pass the PIL image
                        logger.info("Passing PIL color image for 'auto' texture type to utility function.")
                    else:
                        # This case should ideally not be reached if pil_color_image is always loaded upstream
                        logger.warning("PIL color image not available when 'auto' texture type was specified. Proceeding without image.")

                logger.info(f"Using texture type '{texture_type}' for mask {mask_key}")
                
                # Use the new function to generate the heightmap based on texture type
                texture_detail_np = generate_heightmap_by_texture_type(
                    current_mask_bool,
                    texture_type=texture_type,
                    feature_size=feature_size,
                    height=1.0,
                    image=image_param_for_texture_type
                )
                
                final_heightmap_np[current_mask_bool] += detail_scale * texture_detail_np[current_mask_bool]
                logger.debug(f"Applied {texture_type} detail for mask {mask_key}")

            except Exception as e:
                logger.error(f"Error processing mask {mask_path} for detail merging: {e}", exc_info=True)
                # return jsonify({"error": f"Error processing mask {mask_key}: {e}"}), 500

        min_h, max_h = np.min(final_heightmap_np), np.max(final_heightmap_np)
        if max_h > min_h:
             final_heightmap_np = (final_heightmap_np - min_h) / (max_h - min_h)

        merged_heightmap_filename = f"merged_depth_{str(uuid.uuid4())}.png"
        merged_heightmap_path = os.path.join(config.UPLOAD_FOLDER, merged_heightmap_filename)
        files_to_remove.append(merged_heightmap_path)
        try:
            final_heightmap_uint8 = (final_heightmap_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(merged_heightmap_path, final_heightmap_uint8)
            logger.info(f"Saved final merged heightmap to: {merged_heightmap_path}")
            heightmap_to_use = merged_heightmap_path
        except Exception as e:
            logger.error(f"Error saving merged heightmap to {merged_heightmap_path}: {e}", exc_info=True)
            return jsonify({"error": f"Error saving merged heightmap: {e}"}), 500
    else:
        heightmap_to_use = heightmap_path

    # --- 5) Retrieve 3D Model Parameters ---
    try:
        block_width = float(request.form.get("block_width", 100))
        block_length = float(request.form.get("block_length", 100))
        block_thickness = float(request.form.get("block_thickness", 10))
        depth = float(request.form.get("depth", 5))
        base_height = float(request.form.get("base_height", 0))
        mode = request.form.get("mode", "protrude")
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
        color_reference_param = color_image_path
        output_filename = str(uuid.uuid4()) + ".ply"
    else:
        file_type = "stl"
        color_reference_param = None
        output_filename = str(uuid.uuid4()) + ".stl"
    output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
    output_heightmap_filename = os.path.splitext(output_filename)[0] + "_heightmap.png"
    output_heightmap_path = os.path.join(config.OUTPUT_FOLDER, output_heightmap_filename)
    logger.info(f"Preparing to generate {file_type.upper()} model. Output: {output_path}")

    # --- 7) Generate the 3D Model ---
    try:
        logger.info(f"Generating 3D model using heightmap: {heightmap_to_use}")
        # Use imported function
        generate_block_from_heightmap(
            heightmap_path=heightmap_to_use,
            output_path=output_path,
            block_width=block_width,
            block_length=block_length,
            block_thickness=block_thickness,
            depth=depth,
            base_height=base_height,
            mode=mode,
            invert=invert,
            color_reference=color_reference_param,
        )
        logger.info(f"Successfully generated 3D model: {output_path}")
    except TypeError as te:
         if 'mask_paths' in str(te) or 'mask_metadata' in str(te):
              logger.error(f"Mismatch calling generation function. Error: {te}", exc_info=True)
              # TODO: Clean up files
              return jsonify({"error": "Internal configuration error: 3D generation function needs update."}), 500
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
        shutil.copy2(heightmap_to_use, output_heightmap_path)
    except Exception as e:
        logger.error(f"Error copying final heightmap to output folder: {e}", exc_info=True)

    logger.info(f"Cleaning up {len(files_to_remove)} temporary files...")
    for f_path in files_to_remove:
        try:
            if f_path and os.path.exists(f_path):
                os.remove(f_path)
                logger.debug(f"Cleaned up temporary file: {f_path}")
        except OSError as e:
            logger.warning(f"Could not clean up temporary file {f_path}: {e}")

    # --- 9) Return Success Response ---
    # In development (running directly with Flask), use full URL. In production (Docker), use relative path
    if config.FLASK_ENV == "development":
        base_url = request.host_url.rstrip('/')
        file_url = f"{base_url}/outputs/{output_filename}"
        heightmap_file_url = f"{base_url}/outputs/{output_heightmap_filename}"
    else:
        file_url = f"/outputs/{output_filename}"
        heightmap_file_url = f"/outputs/{output_heightmap_filename}"
    
    logger.info(f"Returning success response. Model URL: {file_url}, Heightmap URL: {heightmap_file_url}")
    return jsonify({
        "fileUrl": file_url,
        "fileType": file_type,
        "heightmapFileUrl": heightmap_file_url
    })
