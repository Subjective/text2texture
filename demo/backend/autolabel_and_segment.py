import argparse
import os
import sys
import numpy as np
import torch
import cv2 # Using cv2 for image loading/saving, can use PIL as well
from PIL import Image
from dotenv import load_dotenv # Import dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file (for SAM_HQ_CHECKPOINT_PATH)

# --- Model Imports (Requires 'pip install segment-anything-hq') ---
try:
    # SAM-HQ Imports from the installed package
    from segment_anything_hq import sam_model_registry, SamPredictor

    # Florence-2 Imports
    from transformers import AutoModelForCausalLM, AutoProcessor

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'segment-anything-hq', 'transformers', and 'python-dotenv' are installed.")
    print("Install SAM-HQ using: pip install segment-anything-hq")
    print("Install transformers using: pip install transformers accelerate")
    print("Install dotenv using: pip install python-dotenv")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Utility Functions ---
def load_image(image_path):
    """Loads an image using OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_masks(masks, labels, scores, output_dir, image_filename):
    """Saves masks as PNG files and labels/scores in a text file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.splitext(image_filename)[0]
    info_lines = []

    # Ensure masks, labels, and scores align
    num_masks = masks.shape[0]
    num_labels = len(labels)
    num_scores = len(scores) if isinstance(scores, (list, np.ndarray, torch.Tensor)) else 0


    if num_labels != num_masks or num_scores != num_masks:
         print(f"Warning: Mismatch in number of items - Masks: {num_masks}, Labels: {num_labels}, Scores: {num_scores}.")
         # Adjust labels/scores, prioritizing alignment with masks
         min_count = num_masks # Save based on number of masks generated
         labels = labels[:min_count] if num_labels >= min_count else labels + ['unknown'] * (min_count - num_labels)
         if isinstance(scores, (np.ndarray, torch.Tensor)):
             # Ensure scores tensor has the same device as masks if possible, otherwise use CPU for zeros
             zeros_device = scores.device if isinstance(scores, torch.Tensor) else torch.device('cpu')
             scores = scores[:min_count] if num_scores >= min_count else torch.cat((scores, torch.zeros(min_count - num_scores, device=zeros_device))) if isinstance(scores, torch.Tensor) else np.concatenate((scores, np.zeros(min_count - num_scores)))
         elif isinstance(scores, list):
             scores = scores[:min_count] if num_scores >= min_count else scores + [0.0] * (min_count - num_scores)
         else: # Handle case where scores might be None or unexpected type
             scores = [0.0] * min_count
         print(f"Adjusted counts - Masks: {min_count}, Labels: {len(labels)}, Scores: {len(scores)}")


    # Ensure scores is iterable (list or 1D tensor/array)
    if isinstance(scores, torch.Tensor): scores_list = scores.squeeze().cpu().tolist() # Move to CPU before converting
    elif isinstance(scores, np.ndarray): scores_list = scores.squeeze().tolist()
    elif isinstance(scores, list): scores_list = scores
    else: scores_list = [scores] * num_masks # Fallback if single score or None

    if len(scores_list) != num_masks: # Final check after squeeze
        print(f"Warning: Final score list length ({len(scores_list)}) doesn't match mask count ({num_masks}). Using 0.0 for scores.")
        scores_list = [0.0] * num_masks


    for i, (mask, label, score) in enumerate(zip(masks, labels, scores_list)):
        # mask shape should be (H, W) or (1, H, W)
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255 # Convert boolean tensor to uint8 mask
        safe_label = ''.join(c if c.isalnum() else '_' for c in label) # Sanitize label for filename
        mask_filename = f"{base_filename}_mask_{i}_{safe_label}.png"
        mask_filepath = os.path.join(output_dir, mask_filename)
        try:
            # Ensure mask_np is writeable (sometimes tensors from GPU aren't directly)
            mask_np_writeable = np.require(mask_np, requirements=['W'])
            cv2.imwrite(mask_filepath, mask_np_writeable)
            info_lines.append(f"{mask_filename}: label='{label}', score={score:.4f}")
        except Exception as e:
            print(f"Error saving mask {i} ('{label}') to {mask_filepath}: {e}")


    info_filepath = os.path.join(output_dir, f"{base_filename}_labels.txt")
    with open(info_filepath, 'w') as f:
        f.write("\n".join(info_lines))
    print(f"Saved masks and labels to {output_dir}")


# --- Device Selection ---
def get_device(requested_device: str | None = None) -> torch.device:
    """Selects the appropriate torch device based on availability and request."""
    if requested_device:
        if requested_device == "cuda" and torch.cuda.is_available():
            print("Using requested device: CUDA")
            return torch.device("cuda")
        # Check for MPS availability specifically for MacOS
        if requested_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Using requested device: MPS")
            # Enable fallback for operations not fully supported on MPS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("\nWarning: Support for MPS devices is preliminary. "
                  "It might give numerically different outputs and sometimes degraded performance.\n")
            return torch.device("mps")
        if requested_device == "cpu":
            print("Using requested device: CPU")
            return torch.device("cpu")
        print(f"Warning: Requested device '{requested_device}' not available or recognized. Detecting automatically.")

    # Automatic detection
    if torch.cuda.is_available():
        print("CUDA available. Using CUDA.")
        return torch.device("cuda")
    # Check for MPS availability specifically for MacOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS available. Using MPS.")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("\nWarning: Support for MPS devices is preliminary. "
              "It might give numerically different outputs and sometimes degraded performance.\n")
        return torch.device("mps")
    print("CUDA and MPS not available. Using CPU.")
    return torch.device("cpu")


# --- Main Processing Function ---
def autolabel_and_segment(image_path, output_dir, args, device: torch.device):
    """
    Performs automatic labeling and segmentation using Florence-2 and SAM-HQ.
    """
    # device is now passed as an argument
    image_filename = os.path.basename(image_path)
    print(f"Processing {image_filename} on device {device}...")

    # 1. Load Image
    try:
        image_rgb = load_image(image_path)
        image_pil = Image.fromarray(image_rgb) # Florence-2 might prefer PIL
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


    # 2. Load Models
    print("Loading models...")
    # Load Florence-2
    try:
        print(f"Loading Florence-2 model: {args.florence2_model_id}...")
        # Load model directly to the determined device
        florence_model = AutoModelForCausalLM.from_pretrained(args.florence2_model_id, trust_remote_code=True).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained(args.florence2_model_id, trust_remote_code=True)
        print("Florence-2 model loaded.")
    except Exception as e:
        print(f"Error loading Florence-2 model '{args.florence2_model_id}': {e}")
        print("Ensure the model ID is correct, transformers/accelerate are installed, and you have internet connectivity.")
        sys.exit(1)

    # Load SAM-HQ using model type and checkpoint - **MODIFIED LOADING**
    try:
        print(f"Loading SAM-HQ model type: {args.sam_model_type} structure...")
        if not args.sam_checkpoint or not os.path.exists(args.sam_checkpoint):
             raise FileNotFoundError(f"SAM checkpoint file not found or not specified. Path: '{args.sam_checkpoint}'")

        # Step 1: Instantiate model structure (pass checkpoint=None if supported, else dummy path might be needed by registry)
        # Let's try checkpoint=None first, as it's cleaner if it works.
        # NOTE: If this line fails because checkpoint=None is invalid for the registry,
        # you might need to pass args.sam_checkpoint here and then overwrite the state_dict below.
        sam_model = sam_model_registry[args.sam_model_type](checkpoint=None)

        print(f"Loading checkpoint state_dict: {args.sam_checkpoint} with map_location='{device}'...")
        # Step 2: Load state dict with map_location explicitly
        state_dict = torch.load(args.sam_checkpoint, map_location=device)

        # Step 3: Load the state dict into the model structure
        sam_model.load_state_dict(state_dict)

        # Step 4: Ensure model is on the correct device and in eval mode
        sam_model = sam_model.to(device).eval()

        sam_predictor = SamPredictor(sam_model)
        print(f"SAM-HQ model '{args.sam_model_type}' loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the SAM checkpoint path is correct (provided via --sam_checkpoint or SAM_HQ_CHECKPOINT_PATH in .env).")
        sys.exit(1)
    except KeyError:
         print(f"Error: SAM model type '{args.sam_model_type}' not found in registry.")
         sys.exit(1)
    except TypeError as te:
         # Handle specific TypeError if checkpoint=None is not allowed by registry
         if "'NoneType'" in str(te):
              print(f"\nError: Failed loading model structure with checkpoint=None ({te}).")
              print("The sam_model_registry might require a valid checkpoint path during instantiation.")
              print("Attempting workaround: Loading structure with the path, then reloading state_dict with map_location.")
              try:
                  # WORKAROUND: Instantiate with the checkpoint path, then immediately overwrite state_dict
                  sam_model = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint) # Initial load (might load to wrong device)
                  state_dict = torch.load(args.sam_checkpoint, map_location=device) # Load correctly mapped state_dict
                  sam_model.load_state_dict(state_dict) # Overwrite
                  sam_model = sam_model.to(device).eval() # Ensure final move and eval mode
                  sam_predictor = SamPredictor(sam_model)
                  print(f"SAM-HQ model '{args.sam_model_type}' loaded successfully using workaround.")
              except Exception as e_workaround:
                  print(f"Workaround failed: {e_workaround}")
                  print("Check the model type, checkpoint compatibility, and dependencies (segment-anything-hq installed?).")
                  sys.exit(1)
         else:
            # Re-raise other TypeErrors
            print(f"Error loading SAM-HQ model (TypeError): {te}")
            print("Check the model type, checkpoint compatibility, and dependencies (segment-anything-hq installed?).")
            sys.exit(1)
    except Exception as e:
        # Catch-all for other loading errors (like the original CUDA error if map_location didn't fix it)
        print(f"Error loading SAM-HQ model: {e}")
        print("Check the model type, checkpoint compatibility, and dependencies (segment-anything-hq installed?).")
        sys.exit(1)


    print("Models loaded successfully.")

    # 3. Florence-2 Object Detection / Labeling
    print("Running Florence-2 for object detection/labeling...")
    od_prompt = "<OD>"
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
        print(f"Florence-2 raw output structure: {parsed_answer}") # Debug print

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
            detected_boxes_xyxy = torch.empty((0, 4), device=device) # Keep on correct device
            detected_labels = []
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
        sys.exit(1)

    # 4. SAM-HQ Segmentation (using predict_torch for batching)
    if detected_boxes_xyxy.numel() == 0:
        print("No objects detected by Florence-2. Skipping segmentation.")
        return

    print(f"Running SAM-HQ segmentation (hq_token_only={args.hq_token_only})...")
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
            hq_token_only=args.hq_token_only, # Use HQ output token if flag is set
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
        sys.exit(1)


    # 5. Save Results (masks and scores are moved to CPU inside save_masks)
    save_masks(final_masks, detected_labels, final_scores, output_dir, image_filename)
    print(f"Processing for {image_filename} finished.")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Object Labeling and Segmentation using Florence-2 and SAM-HQ (via pip package)")

    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--output_dir", default="./output_autolabel_samhq_pip", help="Directory to save masks and labels.")

    # --- Model Configs & Paths ---
    parser.add_argument(
        "--florence2_model_id",
        default="microsoft/Florence-2-large-ft",
        help="Hugging Face model ID for Florence-2 (e.g., microsoft/Florence-2-large-ft)."
    )
    parser.add_argument(
        "--sam_model_type",
        default="vit_h",
        choices=['vit_b', 'vit_l', 'vit_h', 'vit_tiny'],
        help="Type of SAM-HQ model to load (e.g., 'vit_l')."
    )
    # Checkpoint path: Not required, can be loaded from .env
    parser.add_argument(
        "--sam_checkpoint",
        default=None, # Default to None, will check env var later
        help="Path to the SAM-HQ model checkpoint (.pth file). Overrides SAM_HQ_CHECKPOINT_PATH from .env if provided."
    )

    # --- Other parameters ---
    parser.add_argument(
        "--device",
        default=None, # Default to None for auto-detection
        choices=['cuda', 'mps', 'cpu'],
        help="Force device to use ('cuda', 'mps', 'cpu'). Detects automatically if not specified."
        )
    parser.add_argument(
        "--hq_token_only",
        action='store_true',
        help="Use only the HQ output token for prediction (recommended for single objects). Default is False (combines SAM and HQ output)."
        )

    args = parser.parse_args()

    # --- Determine Device ---
    selected_device = get_device(args.device)

    # --- Resolve Checkpoint Path ---
    if args.sam_checkpoint is None:
        # Try loading from environment variable if not provided via command line
        args.sam_checkpoint = os.getenv("SAM_HQ_CHECKPOINT_PATH")
        if args.sam_checkpoint:
            print(f"Using SAM checkpoint from environment variable: {args.sam_checkpoint}")
        else:
            print("Error: SAM checkpoint path not provided via --sam_checkpoint argument or SAM_HQ_CHECKPOINT_PATH environment variable.")
            print("Please provide the path using either method.")
            sys.exit(1)
    else:
        # Checkpoint was provided via command line argument
        print(f"Using SAM checkpoint from command line argument: {args.sam_checkpoint}")


    # --- Basic Validation ---
    if not os.path.exists(args.image_path):
        print(f"Error: Input image not found at {args.image_path}")
        sys.exit(1)
    # Checkpoint existence is checked again here after potentially loading from env
    if not args.sam_checkpoint or not os.path.exists(args.sam_checkpoint):
         print(f"Error: SAM checkpoint file not found at the resolved path: {args.sam_checkpoint}")
         sys.exit(1)


    # --- Run ---
    autolabel_and_segment(args.image_path, args.output_dir, args, selected_device)