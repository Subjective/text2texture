# Stage 1: Clone SAM2 repository
FROM python:3.10-slim AS sam2_cloner
WORKDIR /sam2_src
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/facebookresearch/sam2.git .

# Stage 2: Build the main backend image
FROM python:3.10-slim
LABEL maintainer="Joshua"
LABEL description="Backend service for Text2Texture Demo"

WORKDIR /app

# Install system dependencies (git needed for sam2 install, potentially others for CV libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    # Add any other system dependencies required by packages in requirements.txt
    && rm -rf /var/lib/apt/lists/*

# Set environment variables (can be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    # Set paths assuming volumes will be mounted at /app/checkpoints and /app/configs
    SAM2_CONFIG_PATH=/app/configs/sam2.1/sam2.1_hiera_l.yaml \
    SAM2_CHECKPOINT_PATH=/app/checkpoints/sam2.1_hiera_large.pt \
    SAM_HQ_CHECKPOINT_PATH=/app/checkpoints/sam_hq_vit_l.pth \
    # Ensure UPLOAD_FOLDER and OUTPUT_FOLDER are within /app if not mounted externally
    UPLOAD_FOLDER=/app/uploads \
    OUTPUT_FOLDER=/app/outputs

# Create necessary directories (volumes will overwrite these if mounted)
RUN mkdir -p $UPLOAD_FOLDER $OUTPUT_FOLDER /app/checkpoints /app/configs

# Copy SAM2 source from the first stage
COPY --from=sam2_cloner /sam2_src /app/sam2_src

# Copy requirements first for layer caching
COPY requirements.txt .

# Upgrade pip and install requirements
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 from the cloned source
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -e ./sam2_src

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]