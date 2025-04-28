import os
import sys
import numpy as np
import torch
import cv2 # Using cv2 for image loading/saving, can use PIL as well
from PIL import Image
from typing import Tuple, List

# --- Model Imports (Requires 'pip install segment-anything-hq') ---
# These imports are needed by the functions using the models
try:
    # SAM-HQ Imports from the installed package
    from segment_anything_hq import SamPredictor # Only need predictor here
    # Florence-2 Imports
    from transformers import AutoModelForCausalLM, AutoProcessor # Keep these for type hinting if needed
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'segment-anything-hq' and 'transformers' are installed.")
    print("Install SAM-HQ using: pip install segment-anything-hq")
    print("Install transformers using: pip install transformers accelerate")
    # Don't exit here, let the calling application handle it if models can't be loaded
    # sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    # sys.exit(1)


# --- Main Processing Function ---
def generate_masks_from_image(
    image_rgb: np.ndarray,
    florence_model, # Pass loaded model
    florence_processor, # Pass loaded processor
    sam_predictor: SamPredictor, # Pass loaded predictor instance
    device: torch.device,
    hq_token_only: bool = False
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Performs automatic labeling and segmentation using pre-loaded Florence-2 and SAM-HQ models.

    Args:
        image_rgb: Input image as a NumPy array (H, W, 3) in RGB format.
        florence_model: Pre-loaded Florence-2 model instance.
        florence_processor: Pre-loaded Florence-2 processor instance.
        sam_predictor: Pre-loaded SAM-HQ predictor instance.
        device: The torch device to run inference on.
        hq_token_only: Flag for SAM-HQ prediction mode.

    Returns:
        A tuple containing:
        - final_masks: Torch tensor of boolean masks (num_masks, H, W) on the specified device.
        - detected_labels: List of string labels corresponding to the masks.
        - final_scores: Torch tensor of confidence scores (num_masks,) on the specified device.
        Returns empty tensors/list if no objects are detected or an error occurs.
    """
    print(f"Generating masks for image of shape {image_rgb.shape} on device {device}...")

    # 1. Prepare Image for Florence-2
    try:
        image_pil = Image.fromarray(image_rgb) # Florence-2 prefers PIL
    except Exception as e:
        print(f"Error converting NumPy image to PIL: {e}")
        # Return empty results on error
        return torch.empty((0, *image_rgb.shape[:2]), device=device), [], torch.empty((0,), device=device)

    # 2. Florence-2 Object Detection / Labeling
    print("Running Florence-2 for object detection/labeling...")
    od_prompt = "<OD>"
    detected_boxes_xyxy = torch.empty((0, 4), device=device) # Initialize empty on correct device
    detected_labels = []
    try:
        # Ensure inputs are on the correct device
        inputs = florence_processor(text=od_prompt, images=image_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            # generated_text is on CPU after decode
            generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse Florence-2 output
        image_width, image_height = image_pil.size
        parsed_answer = florence_processor.post_process_generation(
            generated_text,
            task=od_prompt,
            image_size=(image_width, image_height)
        )
        # print(f"Florence-2 raw output structure: {parsed_answer}") # Debug print

        # Extract boxes and labels
        if od_prompt not in parsed_answer or not isinstance(parsed_answer[od_prompt], dict):
             raise ValueError(f"Unexpected Florence-2 output format. Task prompt '{od_prompt}' not found or not a dict.")
        od_results = parsed_answer[od_prompt]
        if 'bboxes' not in od_results or 'labels' not in od_results:
            raise ValueError("Florence-2 output dict does not contain 'bboxes' or 'labels'.")
        bboxes_raw = od_results['bboxes']
        labels_raw = od_results['labels']
        if not isinstance(bboxes_raw, list) or not isinstance(labels_raw, list):
             raise ValueError("'bboxes' and 'labels' should be lists.")

        if not bboxes_raw or not labels_raw:
            print("Warning: Florence-2 did not detect any objects.")
            # Keep detected_boxes_xyxy and detected_labels as empty
        elif len(bboxes_raw) != len(labels_raw):
            print(f"Warning: Mismatch between number of boxes ({len(bboxes_raw)}) and labels ({len(labels_raw)}). Using minimum count.")
            min_len = min(len(bboxes_raw), len(labels_raw))
            bboxes_raw = bboxes_raw[:min_len]
            labels_raw = labels_raw[:min_len]
            detected_boxes_xyxy = torch.tensor(bboxes_raw, device=device, dtype=torch.float)
            detected_labels = labels_raw
            print(f"Florence-2 detected {len(detected_labels)} objects (truncated): {detected_labels}")
        else:
             # Ensure tensor is created on the correct device
             detected_boxes_xyxy = torch.tensor(bboxes_raw, device=device, dtype=torch.float)
             detected_labels = labels_raw
             print(f"Florence-2 detected {len(detected_labels)} objects: {detected_labels}")

    except Exception as e:
        print(f"Error running Florence-2 or parsing its output: {e}")
        print("Check the Florence-2 task prompt, output format, and parsing logic above.")
        print("Raw generated text:", generated_text if 'generated_text' in locals() else "N/A")
        print("Parsed answer:", parsed_answer if 'parsed_answer' in locals() else "N/A")
        # Return empty results on error
        return torch.empty((0, *image_rgb.shape[:2]), device=device), [], torch.empty((0,), device=device)

    # 3. SAM-HQ Segmentation (using predict_torch for batching)
    if detected_boxes_xyxy.numel() == 0:
        print("No objects detected by Florence-2. Skipping segmentation.")
        # Return empty results as initialized
        return detected_boxes_xyxy, detected_labels, torch.empty((0,), device=device)

    print(f"Running SAM-HQ segmentation (hq_token_only={hq_token_only})...")
    final_masks = torch.empty((0, *image_rgb.shape[:2]), device=device) # Initialize empty
    final_scores = torch.empty((0,), device=device) # Initialize empty
    try:
        # Set image for SAM Predictor (predictor internally handles device transfer if needed)
        sam_predictor.set_image(image_rgb)

        # Prepare prompts for SAM-HQ (bounding boxes in XYXY format)
        # predict_torch expects torch tensors and *transformed* boxes
        input_boxes_torch = detected_boxes_xyxy # Already on the correct device
        # Transform boxes to the model's input scale
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes_torch, image_rgb.shape[:2])

        # Predict masks using predict_torch for batched input
        # Ensure model and inputs are on the same device (should be handled by predictor and previous steps)
        masks, scores, logits = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes, # Use transformed boxes
            multimask_output=False, # Get single best mask per box prompt
            hq_token_only=hq_token_only, # Use HQ output token if flag is set
        )
        # masks shape: (num_boxes, 1, H, W) - torch tensor (bool) on device
        # scores shape: (num_boxes, 1) - torch tensor (float) on device

        # Squeeze the mask dimension (C=1) and scores dimension
        final_masks = masks.squeeze(1) # Shape: (num_boxes, H, W) still on device
        final_scores = scores.squeeze(1) # Shape: (num_boxes,) still on device

        print(f"Generated {final_masks.shape[0]} masks with scores: {final_scores.cpu().tolist()}") # Move scores to CPU for printing

    except Exception as e:
        print(f"Error during SAM-HQ prediction: {e}")
        print("Check input box format and predictor arguments.")
        # Return empty results on error, keep labels from Florence
        return torch.empty((0, *image_rgb.shape[:2]), device=device), detected_labels, torch.empty((0,), device=device)


    # 4. Return Results
    print(f"Mask generation finished. Returning {final_masks.shape[0]} masks.")
    return final_masks, detected_labels, final_scores