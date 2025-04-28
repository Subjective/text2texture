import logging
import numpy as np
import torch

from flask import Blueprint, jsonify, request

# Import configuration, models, processing functions, and utilities
import config
import models
from processing.segmentation import generate_masks_from_image, predict_mask_with_prompt # Use renamed function
from utils.image_utils import decode_base64_image, encode_mask_to_base64_png

# Get logger
logger = logging.getLogger(__name__)

# Define Blueprint
masks_bp = Blueprint('masks', __name__, url_prefix='/api')

# --- API Endpoints ---

@masks_bp.route("/predict_mask", methods=["POST"])
def api_predict_mask():
    """API endpoint to predict mask based on image and points/box using SAM-HQ."""
    sam_hq_predictor = models.get_sam_hq_predictor() # Get HQ-SAM predictor
    if sam_hq_predictor is None:
        logger.error("SAM-HQ model not loaded, cannot predict mask.")
        return jsonify({"status": "error", "message": "Mask prediction model not loaded"}), 503 # Service Unavailable

    data = request.get_json()
    if not data: return jsonify({"status": "error", "message": "No input data provided"}), 400

    base64_image_str = data.get("image")
    points_data = data.get("points") # e.g., [{"x": 100, "y": 200}, ...]
    box_data = data.get("box")       # e.g., [10, 20, 150, 180]

    if not base64_image_str:
        return jsonify({"status": "error", "message": "Missing 'image' in request"}), 400
    if points_data is None and box_data is None:
         return jsonify({"status": "error", "message": "Missing 'points' or 'box' in request"}), 400
    if points_data is not None and box_data is not None:
         return jsonify({"status": "error", "message": "Provide either 'points' or 'box', not both"}), 400

    logger.info("Decoding image for mask prediction...")
    image_rgb = decode_base64_image(base64_image_str) # Use utility function
    if image_rgb is None: return jsonify({"status": "error", "message": "Invalid image data provided"}), 400
    logger.info(f"Image decoded successfully: shape={image_rgb.shape}")

    # Prepare points or box arguments for the processing function
    points_list = None
    point_labels_list = None
    box_list = None
    prompt_type_log = ""

    if points_data is not None:
        prompt_type_log = "points"
        if not isinstance(points_data, list) or len(points_data) == 0:
            return jsonify({"status": "error", "message": "'points' must be a non-empty list of {x, y} objects"}), 400
        try:
            # Convert points from frontend format [{x: val, y: val}, ...] to list of tuples
            # Assuming all points are foreground points (label=1) for now
            points_list = [(p["x"], p["y"]) for p in points_data]
            point_labels_list = [1] * len(points_list) # All foreground points
            logger.info(f"Processed {len(points_list)} points.")
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Invalid point format received: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Invalid point format: {e}"}), 400
    elif box_data is not None:
        prompt_type_log = "box"
        if not isinstance(box_data, list) or len(box_data) != 4 or not all(isinstance(n, (int, float)) for n in box_data):
             return jsonify({"status": "error", "message": "'box' must be a list of 4 numbers [x1, y1, x2, y2]"}), 400
        box_list = [int(coord) for coord in box_data] # Ensure integer coordinates if needed by model/processing
        logger.info(f"Processed box: {box_list}")

    # --- Run SAM-HQ Prediction ---
    try:
        # Use config for device and dtype
        with torch.autocast(device_type=config.DEVICE.type, dtype=config.AUTOCAST_DTYPE):
            logger.info(f"Running SAM-HQ {prompt_type_log} prediction...")
            # Call the updated function from segmentation.py
            masks_tensor, scores_tensor = predict_mask_with_prompt(
                image_rgb=image_rgb,
                points=points_list, # Will be None if using box
                point_labels=point_labels_list, # Will be None if using box
                box=box_list, # Will be None if using points
                sam_predictor=sam_hq_predictor,
                device=config.DEVICE,
                multimask_output=False, # Get single best mask
                hq_token_only=False # Use default setting for now
            )
            logger.info(f"Prediction complete. Mask shape: {masks_tensor.shape}, Scores shape: {scores_tensor.shape}")

            # Process the output mask (should be shape [1, H, W] if multimask=False)
            if isinstance(masks_tensor, torch.Tensor):
                if masks_tensor.ndim == 3 and masks_tensor.shape[0] == 1:
                    output_mask_tensor = masks_tensor.squeeze(0) # Remove mask dim -> [H, W]
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
            scores = scores_tensor # Use the scores tensor directly
    except Exception as e:
        logger.error(f"Error during SAM-HQ {prompt_type_log} prediction: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Mask prediction failed: {e}"}), 500

    # --- Encode Result Mask to Base64 PNG ---
    logger.info("Encoding predicted mask to Base64 PNG...")
    b64_png_mask = encode_mask_to_base64_png(output_mask_bool) # Use utility function
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


@masks_bp.route("/generate_masks", methods=["POST"])
def api_generate_masks():
    """API endpoint for automatic mask generation using Florence-2 and SAM-HQ."""
    # Get models from models module
    florence_model = models.get_florence_model()
    florence_processor = models.get_florence_processor()
    sam_hq_predictor = models.get_sam_hq_predictor()

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
    image_rgb = decode_base64_image(base64_image_str) # Use utility function
    if image_rgb is None:
        return jsonify({"status": "error", "message": "Invalid image data provided"}), 400
    logger.info(f"Image decoded successfully: shape={image_rgb.shape}")

    try:
        logger.info("Calling generate_masks_from_image function...")
        # Assuming default hq_token_only=False for now, could make this configurable via request
        # Use config for device
        masks_tensor, labels, scores_tensor = generate_masks_from_image(
            image_rgb=image_rgb,
            florence_model=florence_model,
            florence_processor=florence_processor,
            sam_predictor=sam_hq_predictor,
            device=config.DEVICE,
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
            b64_png_mask = encode_mask_to_base64_png(mask_bool) # Use utility function
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
