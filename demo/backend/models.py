import logging
import os
import torch
from openai import OpenAI

# Import config values
import config

# --- Model Specific Imports ---

# SAM2 Imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2 library components: {e}")
    print("Ensure SAM2 is installed correctly ('pip install -e ./sam2'). Point-based prediction will be unavailable.")
    build_sam2 = None
    SAM2ImagePredictor = None

# SAM-HQ and Florence-2 Imports
try:
    from segment_anything_hq import sam_model_registry, SamPredictor
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as e:
    print(f"Error importing SAM-HQ/Florence-2 library components: {e}")
    print("Ensure 'segment-anything-hq' and 'transformers' are installed. Auto-mask generation will be unavailable.")
    sam_model_registry = None
    SamPredictor = None
    AutoModelForCausalLM = None
    AutoProcessor = None

# --- Module-level Variables ---

# Logger setup
# Note: This logger will inherit the root logger's configuration
# set up in app.py (or wherever logging is configured).
logger = logging.getLogger(__name__)

# Variables to hold initialized models/clients
_sam2_predictor: SAM2ImagePredictor | None = None
_florence_model = None
_florence_processor = None
_sam_hq_predictor: SamPredictor | None = None
_openai_client: OpenAI | None = None
_zoe_depth_model = None # Variable for ZoeDepth model


# --- Initialization Functions ---

def initialize_openai_client():
    """Initializes and returns the OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        logger.debug("OpenAI client already initialized.")
        return _openai_client # Already initialized

    logger.info("Initializing OpenAI client...")
    if not config.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not found in config or .env file. Text-to-image generation will be disabled.")
        _openai_client = None
    else:
        try:
            _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            # Optional: Test connection (add error handling if needed)
            # _openai_client.models.list()
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            _openai_client = None
    return _openai_client

def load_sam2_model():
    """Loads the SAM2 model predictor using paths from config."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        logger.info("SAM2 predictor already loaded.")
        return True # Already loaded

    if not SAM2ImagePredictor or not build_sam2:
         logger.error("SAM2 library components not imported correctly. Cannot load SAM2 model.")
         return False

    if not os.path.exists(config.SAM2_CONFIG_PATH):
        logger.error(f"SAM2 Config file not found: {config.SAM2_CONFIG_PATH}")
        return False
    if not os.path.exists(config.SAM2_CHECKPOINT_PATH):
        logger.error(f"SAM2 Checkpoint file not found: {config.SAM2_CHECKPOINT_PATH}")
        return False

    try:
        logger.info(f"Building SAM2 base model from {config.SAM2_CONFIG_PATH} and {config.SAM2_CHECKPOINT_PATH}...")
        sam2_model = build_sam2(config.SAM2_CONFIG_PATH, config.SAM2_CHECKPOINT_PATH, device=config.DEVICE)
        logger.info("Initializing SAM2ImagePredictor...")
        _sam2_predictor = SAM2ImagePredictor(sam2_model)
        logger.info(f"SAM2 predictor loaded successfully on device: {config.DEVICE}")
        if config.DEVICE.type == "mps":
             logger.warning( # Use logger instead of print
                "Support for MPS devices is preliminary for SAM2. "
                "It might give numerically different outputs and sometimes degraded performance."
            )
        return True
    except Exception as e:
        logger.error(f"Error loading SAM2 model: {e}", exc_info=True)
        _sam2_predictor = None
        return False

def load_autolabel_models():
    """Loads the Florence-2 and SAM-HQ models using paths from config."""
    global _florence_model, _florence_processor, _sam_hq_predictor

    # Check if already loaded
    if _florence_model and _florence_processor and _sam_hq_predictor:
        logger.info("Autolabel models (Florence-2, SAM-HQ) already loaded.")
        return True

    florence_loaded = False
    sam_hq_loaded = False

    # --- Load Florence-2 ---
    if not _florence_model and AutoModelForCausalLM and AutoProcessor: # Check imports succeeded
        try:
            logger.info(f"Loading Florence-2 model: {config.FLORENCE2_MODEL_ID}...")
            _florence_model = AutoModelForCausalLM.from_pretrained(config.FLORENCE2_MODEL_ID, trust_remote_code=True).to(config.DEVICE).eval()
            _florence_processor = AutoProcessor.from_pretrained(config.FLORENCE2_MODEL_ID, trust_remote_code=True)
            logger.info("Florence-2 model loaded successfully.")
            florence_loaded = True
        except Exception as e:
            logger.error(f"Error loading Florence-2 model '{config.FLORENCE2_MODEL_ID}': {e}", exc_info=True)
            logger.error("Ensure the model ID is correct, transformers/accelerate are installed, and you have internet connectivity.")
            _florence_model = None
            _florence_processor = None
    elif not _florence_model:
         logger.warning("Florence-2 library components not imported correctly. Cannot load Florence-2 model.")


    # --- Load SAM-HQ ---
    if not _sam_hq_predictor and sam_model_registry and SamPredictor: # Check imports succeeded
        if not config.SAM_HQ_CHECKPOINT_PATH or not os.path.exists(config.SAM_HQ_CHECKPOINT_PATH):
            logger.error(f"SAM-HQ checkpoint file not found or not specified in .env (SAM_HQ_CHECKPOINT_PATH). Path: '{config.SAM_HQ_CHECKPOINT_PATH}'")
            _sam_hq_predictor = None
        else:
            try:
                logger.info(f"Loading SAM-HQ model type: {config.SAM_HQ_MODEL_TYPE} structure...")
                sam_model_hq = sam_model_registry[config.SAM_HQ_MODEL_TYPE](checkpoint=None) # Try checkpoint=None first

                logger.info(f"Loading SAM-HQ checkpoint state_dict: {config.SAM_HQ_CHECKPOINT_PATH} with map_location='{config.DEVICE}'...")
                state_dict = torch.load(config.SAM_HQ_CHECKPOINT_PATH, map_location=config.DEVICE)
                sam_model_hq.load_state_dict(state_dict)
                sam_model_hq = sam_model_hq.to(config.DEVICE).eval()
                _sam_hq_predictor = SamPredictor(sam_model_hq)
                logger.info(f"SAM-HQ predictor ('{config.SAM_HQ_MODEL_TYPE}') loaded successfully on device: {config.DEVICE}")
                sam_hq_loaded = True

            except TypeError as te:
                 if "'NoneType'" in str(te):
                      logger.warning(f"Failed loading SAM-HQ structure with checkpoint=None ({te}). Attempting workaround...")
                      try:
                          # WORKAROUND: Instantiate with the checkpoint path, then immediately overwrite state_dict
                          sam_model_hq = sam_model_registry[config.SAM_HQ_MODEL_TYPE](checkpoint=config.SAM_HQ_CHECKPOINT_PATH) # Initial load
                          state_dict = torch.load(config.SAM_HQ_CHECKPOINT_PATH, map_location=config.DEVICE) # Load correctly mapped state_dict
                          sam_model_hq.load_state_dict(state_dict) # Overwrite
                          sam_model_hq = sam_model_hq.to(config.DEVICE).eval() # Ensure final move and eval mode
                          _sam_hq_predictor = SamPredictor(sam_model_hq)
                          logger.info(f"SAM-HQ predictor ('{config.SAM_HQ_MODEL_TYPE}') loaded successfully using workaround.")
                          sam_hq_loaded = True
                      except Exception as e_workaround:
                          logger.error(f"SAM-HQ workaround failed: {e_workaround}", exc_info=True)
                          _sam_hq_predictor = None
                 else:
                    logger.error(f"Error loading SAM-HQ model (TypeError): {te}", exc_info=True)
                    _sam_hq_predictor = None
            except KeyError:
                 logger.error(f"Error: SAM-HQ model type '{config.SAM_HQ_MODEL_TYPE}' not found in registry.")
                 _sam_hq_predictor = None
            except Exception as e:
                logger.error(f"Error loading SAM-HQ model: {e}", exc_info=True)
                _sam_hq_predictor = None
    elif not _sam_hq_predictor:
        logger.warning("SAM-HQ library components not imported correctly. Cannot load SAM-HQ model.")

    # Return True only if both critical models were intended and successfully loaded
    # Modify this logic based on which models are essential for the app to run
    return florence_loaded and sam_hq_loaded

def load_zoe_depth_model():
    """Loads the ZoeDepth model using config device."""
    global _zoe_depth_model
    if _zoe_depth_model is not None:
        logger.info("ZoeDepth model already loaded.")
        return True

    logger.info("Loading ZoeDepth model...")
    try:
        repo = "isl-org/ZoeDepth"
        # Use device from config
        _zoe_depth_model = torch.hub.load(repo, "ZoeD_N", pretrained=True)
        _zoe_depth_model = _zoe_depth_model.to(config.DEVICE)
        logger.info(f"ZoeDepth model loaded successfully on device: {config.DEVICE}")
        return True
    except Exception as e:
        logger.error(f"Error loading ZoeDepth model: {e}", exc_info=True)
        _zoe_depth_model = None
        return False

def load_all_models():
    """Loads all required models and initializes clients."""
    logger.info("--- Starting Model Loading ---")
    sam2_ok = load_sam2_model()
    autolabel_ok = load_autolabel_models()
    zoe_depth_ok = load_zoe_depth_model() # Load ZoeDepth
    openai_ok = initialize_openai_client() is not None
    logger.info("--- Model Loading Attempted ---")
    logger.info(f"SAM2 Loaded: {sam2_ok}")
    logger.info(f"Autolabel Models Loaded: {autolabel_ok}")
    logger.info(f"ZoeDepth Loaded: {zoe_depth_ok}") # Log ZoeDepth status
    logger.info(f"OpenAI Client Initialized: {openai_ok}")
    # Optionally, add checks here to see if critical models failed and raise an error


# --- Accessor Functions ---

def get_openai_client() -> OpenAI | None:
    """Returns the initialized OpenAI client."""
    # Ensure initialization is attempted if not already done
    # if _openai_client is None:
    #     initialize_openai_client()
    return _openai_client

def get_sam2_predictor() -> SAM2ImagePredictor | None:
    """Returns the initialized SAM2 predictor."""
    # Ensure initialization is attempted if not already done
    # if _sam2_predictor is None:
    #     load_sam2_model()
    return _sam2_predictor

def get_florence_model():
    """Returns the initialized Florence-2 model."""
    # Ensure initialization is attempted if not already done
    # if _florence_model is None:
    #     load_autolabel_models()
    return _florence_model

def get_florence_processor():
    """Returns the initialized Florence-2 processor."""
    # Ensure initialization is attempted if not already done
    # if _florence_processor is None:
    #     load_autolabel_models()
    return _florence_processor

def get_sam_hq_predictor() -> SamPredictor | None:
    """Returns the initialized SAM-HQ predictor."""
    # Ensure initialization is attempted if not already done
    # if _sam_hq_predictor is None:
    #     load_autolabel_models()
    return _sam_hq_predictor

def get_zoe_depth_model():
    """Returns the initialized ZoeDepth model."""
    # Ensure initialization is attempted if not already done
    # if _zoe_depth_model is None:
    #     load_zoe_depth_model() # Or rely on load_all_models during startup
    return _zoe_depth_model