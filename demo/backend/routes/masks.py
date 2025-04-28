import logging
import numpy as np
import torch

from flask import Blueprint, jsonify, request

# Import configuration, models, processing functions, and utilities
import config
import models
from processing.segmentation import generate_masks_from_image # Assuming this is the correct function
from utils.image_utils import decode_base64_image, encode_mask_to_base64_png

# Get logger
logger = logging.getLogger(__name__)

# Define Blueprint
masks_bp = Blueprint('masks', __name__, url_prefix='/api')

# --- API Endpoints ---

@masks_bp.route("/predict_mask", methods=["POST"])
def api_predict_mask():
    """API endpoint to predict mask based on image and points using SAM2."""
    sam2_predictor = models.get_sam2_predictor() # Get predictor from models module
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
    image_rgb = decode_base64_image(base64_image_str) # Use utility function
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
        # Use config for device and dtype
        with torch.autocast(device_type=config.DEVICE.type, dtype=config.AUTOCAST_DTYPE):
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
