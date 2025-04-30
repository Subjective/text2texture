# Configuration settings for the Flask backend application
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Environment ---
FLASK_ENV = os.getenv("FLASK_ENV", "development")  # Defaults to development unless set otherwise

# --- Folders ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
# Ensure folders exist (optional here, could be done in app setup)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Model Paths ---
# SAM-HQ Paths (Update these paths as needed, loaded from .env)
SAM_HQ_MODEL_TYPE = os.getenv("SAM_HQ_MODEL_TYPE", "vit_h") # e.g., vit_h, vit_l
SAM_HQ_CHECKPOINT_PATH = os.getenv("SAM_HQ_CHECKPOINT_PATH") # Must be set in .env

# Florence-2 Path (Hugging Face model ID)
FLORENCE2_MODEL_ID = os.getenv("FLORENCE2_MODEL_ID", "microsoft/Florence-2-large-ft")

# --- Device Selection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    # MPS support is experimental for SAM-HQ
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Autocast Dtype (for SAM-HQ) ---
if DEVICE.type == "cuda":
    AUTOCAST_DTYPE = torch.bfloat16
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    AUTOCAST_DTYPE = torch.float32

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Flask Server Configuration ---
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
# Debug mode should be disabled in production
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"] if FLASK_ENV == "development" else False

# --- Logging ---
# Basic logging level, can be configured further in app setup
LOG_LEVEL = "INFO"

# --- Function Specific Defaults ---
# Could move defaults from function signatures here if desired
# E.g., DEFAULT_CHECKER_SIZE = 8
