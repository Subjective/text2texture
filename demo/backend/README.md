# Demo Backend README

This directory contains the Python Flask backend for the Text2Texture demo application.

## Functionality

The backend provides APIs for:

1.  **Text-to-Image Generation:** Using OpenAI's DALL-E 3 via the `/api/generate_image` endpoint.
2.  **Automatic Mask Generation:** Using Florence-2 for object detection and SAM-HQ for segmentation via the `/api/generate_masks` endpoint.
3.  **Point-based Mask Prediction:** Using SAM 2 via the `/api/predict_mask` endpoint.
4.  **3D Model Generation:** Combining an input image (and optional masks) to generate a heightmap (using ZoeDepth) and then converting it into a 3D model (STL or PLY with color) via the `/api/generate_model` endpoint.
5.  **Serving Output Files:** Generated models and heightmaps are served from the `/outputs/` directory.

## Project Structure

```
demo/backend/
├── app.py              # Flask application setup, API routes, model loading orchestration
├── processing/         # Core image/3D processing logic modules
│   ├── __init__.py
│   ├── segmentation.py   # Auto-labeling (Florence-2) and segmentation (SAM-HQ) logic
│   ├── depth.py          # Depth map generation (ZoeDepth) logic
│   └── conversion_3d.py  # Heightmap-to-3D model (STL/PLY) conversion logic
├── utils/              # Shared utility functions
│   ├── __init__.py
│   ├── visualization.py  # Colormap visualization helper (e.g., colorize)
│   └── mesh_io.py        # Mesh input/output helpers (e.g., write_ply)
├── uploads/            # Temporary storage for uploaded images/masks (created automatically)
├── outputs/            # Storage for generated 3D models and heightmaps (created automatically)
├── requirements.txt    # Python dependencies
├── .mise.toml          # Mise configuration for environment management
├── .env.example        # Example environment variables file
└── README.md           # This file
```

## Setup

1.  **Clone Repositories:**
    *   Ensure you have cloned the main `text2texture` repository.
    *   The `sam2` dependency requires cloning its repository. The `requirements.txt` attempts to install it via `pip install git+...`, but manual cloning might be needed if issues arise.

2.  **Environment & Dependencies (Using Mise - Recommended):**
    *   If you have `mise` installed (see [mise.jdx.dev](https://mise.jdx.dev/)), simply navigate to the `demo/backend` directory in your terminal and run:
        ```bash
        mise install
        # or just `mise i`
        ```
    *   This command will automatically:
        *   Read the `.mise.toml` file.
        *   Install the correct Python version specified.
        *   Create a virtual environment (usually in `.venv`).
        *   Install all dependencies listed in `requirements.txt` into that virtual environment.
    *   `mise` will automatically activate the environment whenever you `cd` into this directory.

3.  **Environment & Dependencies (Manual):**
    *   If not using `mise`, create a virtual environment manually:
        ```bash
        python -m venv .venv # Create venv named .venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   *Note:* This installs PyTorch based on your system's CUDA/MPS/CPU capabilities if available. If you encounter PyTorch installation issues, visit [pytorch.org](https://pytorch.org/) for specific installation commands for your OS and hardware.

4.  **Download Models:**
    *   **SAM 2 Checkpoint:** Download a SAM 2 checkpoint (e.g., `sam2.1_hiera_large.pt`) and place it in a known location (e.g., `../../checkpoints/` relative to this directory).
    *   **SAM-HQ Checkpoint:** Download a SAM-HQ checkpoint (e.g., `sam_hq_vit_h.pth`) and place it in a known location (e.g., `../../checkpoints/`).
    *   **Florence-2 & ZoeDepth:** These are downloaded automatically via `transformers` and `torch.hub` respectively when the application starts, provided you have an internet connection.

5.  **Configure Environment Variables:**
    *   Copy `.env.example` to `.env`.
    *   Edit the `.env` file and fill in the required values:
        *   `OPENAI_API_KEY`: Your API key from OpenAI for DALL-E image generation.
        *   `SAM2_CONFIG_PATH`: Path to the SAM 2 model configuration file (e.g., `../sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml` relative to this backend dir, or an absolute path). Adjust if your `sam2` clone is elsewhere.
        *   `SAM2_CHECKPOINT_PATH`: Path to the downloaded SAM 2 checkpoint file (e.g., `../../checkpoints/sam2.1_hiera_large.pt`). Adjust if your checkpoints are elsewhere.
        *   `SAM_HQ_MODEL_TYPE`: The type of SAM-HQ model used (e.g., `vit_h`, matching the checkpoint).
        *   `SAM_HQ_CHECKPOINT_PATH`: Path to the downloaded SAM-HQ checkpoint file (e.g., `../../checkpoints/sam_hq_vit_h.pth`). Adjust if your checkpoints are elsewhere.
        *   `FLORENCE2_MODEL_ID`: (Optional) Override the default Florence-2 model ID if needed.
        *   `FLASK_HOST`: (Optional) Host to run the server on (default: `127.0.0.1`).
        *   `FLASK_PORT`: (Optional) Port to run the server on (default: `5000`).
        *   `FLASK_DEBUG`: (Optional) Set to `True` or `False` for Flask debug mode (default: `True`).

## Running the Backend

Navigate to the `demo/backend` directory in your terminal.

*   If using `mise`, the environment should be active automatically.
*   If using manual venv, ensure it's activated (`source .venv/bin/activate`).

Then run:

```bash
python app.py
```

The server will start, load the necessary models (which may take some time on the first run), and listen for requests on the configured host and port (e.g., `http://127.0.0.1:5000`).