# Text2Texture

Text2Texture is a tool that allows you to generate 3D textures and models from text prompts. By combining AI-powered image generation, segmentation, and depth estimation, this project provides an end-to-end pipeline for creating detailed 3D assets from simple text descriptions.

## Features

- **Text-to-Image Generation**: Generate images from text prompts using OpenAI's DALL-E 3
- **Automatic Segmentation**: Identify and segment objects in generated images using Florence-2 and SAM-HQ
- **3D Model Generation**: Create 3D models with proper depth using ZoeDepth
- **Interactive Demo**: User-friendly web interface for creating, manipulating, and exporting 3D assets

## Repository Structure

```
text2texture/
├── checkpoints/             # Model checkpoints (created by download script)
├── configs/                 # Configuration files
├── demo/                    # Interactive web-based demo
│   ├── backend/             # Flask backend with AI models
│   │   ├── app.py           # Main Flask application
│   │   ├── requirements.txt # Python dependencies
│   │   └── ...
│   ├── frontend/            # React frontend interface
│   │   ├── src/             # React source code
│   │   └── ...
│   └── docker-compose.yml   # Docker setup for the demo
└── scripts/                 # Utility scripts
    └── download_checkpoints.sh  # Script to download model weights
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for frontend development)
- Docker & Docker Compose (optional, for containerized deployment)
- GPU recommended for faster inference
- OpenAI API key for DALL-E image generation

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Subjective/text2texture.git
   cd text2texture
   ```

2. Download the required model checkpoints:
   ```bash
   bash scripts/download_checkpoints.sh
   ```

3. Set up the demo:

   #### Option 1: Using Docker (recommended for deployment)
   ```bash
   cd demo
   # Create a .env file in demo/backend with your OpenAI API key
   echo "OPENAI_API_KEY=your_openai_api_key" > backend/.env
   # Add other configuration as needed

   # Start the containers
   docker-compose up -d
   ```

   #### Option 2: Manual Setup (recommended for development)

   **Backend**:
   ```bash
   cd demo/backend
   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   # Create .env file with your OpenAI API key and configuration
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   echo "SAM_HQ_CHECKPOINT_PATH=../../checkpoints/sam_hq_vit_h.pth" >> .env
   # Start the backend
   python app.py
   ```

   **Frontend**:
   ```bash
   cd demo/frontend
   npm install
   npm run dev
   ```

4. Access the demo:
   - Using Docker: http://localhost:8080
   - Manual setup: http://localhost:5173 (or the URL shown in terminal)

## Usage

1. Enter a descriptive text prompt to generate an image
2. Use automatic segmentation or select points to create masks for specific areas
3. Adjust parameters for depth and texture generation
4. Generate the 3D model
5. Download the resulting model in STL or PLY format

## Models and Credits

This project uses several state-of-the-art AI models:

- **DALL-E 3** (via OpenAI API) for text-to-image generation
- **Florence-2** (Microsoft) for automatic object detection and labeling
- **SAM-HQ** (High-Quality Segment Anything Model) for precise image segmentation
- **ZoeDepth** (ICCV 2023) for monocular depth estimation

## License

MIT License

Copyright (c) 2023 Text2Texture Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- The SAM-HQ implementation is based on [lkeab/hq-sam](https://github.com/lkeab/hq-sam)
- ZoeDepth implementation is from [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth)
